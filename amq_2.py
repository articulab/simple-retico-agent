"""
ActiveMQ Module
=============

This module defines two incremental modules ZeroMQReader and ZeroMQWriter that act as a
a bridge between ZeroMQ and retico. For this, a ZeroMQIU is defined that contains the
information revceived over the ZeroMQ bridge.
"""

# retico
import os
import json
from collections import deque
import stomp
import json
from urllib.parse import unquote


import retico_core
import retico_core.abstract
from retico_core.abstract import *
from retico_core.log_utils import log_exception


class AMQIU(retico_core.IncrementalUnit):
    """Decorator class for IncrementalUnit that will be sent through ActiveMQ. Adding headers and destination parameters."""

    def __init__(
        self,
        creator=None,
        iuid=0,
        previous_iu=None,
        grounded_in=None,
        decorated_iu=None,
        headers=None,
        destination=None,
        **kwargs,
    ):
        super().__init__(
            creator=creator,
            iuid=iuid,
            previous_iu=previous_iu,
            grounded_in=grounded_in,
        )
        self.decorated_iu = decorated_iu
        self.headers = headers
        self.destination = destination

    @staticmethod
    def type():
        return "AMQ IU"

    def get_deco_iu(self):
        """returns the decorated IU.

        Returns:
            IncrementalUnit: decorated IU.
        """
        return self.decorated_iu

    def set_amq(self, decorated_iu, headers, destination):
        """Sets AMQ IU's parameters.

        Args:
            decorated_iu (IncrementalUnit): decorated IU
            headers (dict): ActiveMQ headers
            destination (str): ActiveMQ destination (topic, queue, etc)
        """
        self.decorated_iu = decorated_iu
        self.headers = headers
        self.destination = destination


class AMQReader(retico_core.AbstractProducingModule):
    """
    Module providing a retico system with ActiveMQ message reception.
    The module can subscribe to different ActiveMQ destinations, each will correspond to a desired IU type.
    At each message reception from one of the destinations, the module transforms the ActiveMQ message to an IncrementalUnit of the corresponding IU type.
    """

    @staticmethod
    def name():
        return "ActiveMQ Reader Module"

    @staticmethod
    def description():
        return "A Module providing a retico system with ActiveMQ message reception"

    @staticmethod
    def output_iu():
        return IncrementalUnit

    def __init__(self, ip, port, **kwargs):
        """Initializes the ActiveMQReader.

        Args:
            ip (str): the IP of the computer.
            port (str): the port corresponding to ActiveMQ
        """
        super().__init__(**kwargs)
        hosts = [(ip, port)]
        self.conn = stomp.Connection(host_and_ports=hosts, auto_content_length=False)
        self.conn.connect("admin", "admin", wait=True)
        self.conn.set_listener("", self.Listener(self))
        self.target_iu_types = dict()

    class Listener(stomp.ConnectionListener):
        """Listener that triggers ANQReader's `on_message` function every time a message is unqueued in one of the subscribed destination."""

        def __init__(self, module):
            super().__init__()
            # in order to use methods of activeMQ we create its instance
            self.module = module

        # Override the methods on_error and on_message provides by the parent class
        def on_error(self, frame):
            self.module.on_listener_error(frame)
            # print('received an error "%s"' % frame.body)

        def on_message(self, frame):
            # self.module.logMessageReception(frame)
            self.module.on_message(frame)

    def add(self, destination, target_iu_type):
        """Subscribe to a new ActiveMQ destination, and stores the desired corresponding IU type in `target_iu_type`.

        Args:
            destination (_type_): the ActiveMQ destination to subscribe to.
            target_iu_type (dict): dictionary of all destination-IU type associations.
        """
        self.conn.subscribe(destination=destination, id=1, ack="auto")
        self.target_iu_types[destination] = target_iu_type

    def on_message(self, frame):
        """The function that is triggered every time a message (= `frame`)is unqueued in one of the subscribed destination.
        The message is then processed an transformed into an IU of the corresponding type.

        Args:
            frame (stomp.frame): the received ActiveMQ message.
        """

        message = frame.body
        destination = frame.headers["destination"]

        if destination not in self.target_iu_types:
            print(destination, "is not a recognized destination")
            return None

        try:
            # try to parse the message to create a dict (it has to be a structured message JSON), and put it in the IU's init parameters.
            # create the decorated IU (cannot use classical create_iu from AbstractModule)
            msg_json = json.loads(message)
            output_iu = self.target_iu_types[destination](
                creator=self,
                iuid=f"{hash(self)}:{self.iu_counter}",
                previous_iu=self._previous_iu,
                grounded_in=None,
                **msg_json,
            )
        except Exception:
            # if message not parsable as a structured message (JSON), then put it as the IU's payload.
            # create the decorated IU (cannot use classical create_iu from AbstractModule)
            output_iu = self.target_iu_types[destination](
                creator=self,
                iuid=f"{hash(self)}:{self.iu_counter}",
                previous_iu=self._previous_iu,
                grounded_in=None,
                payload=message,
            )

        # create the decorated IU (cannot use classical create_iu from AbstractModule)
        # output_iu = self.target_iu_types[destination](
        #     creator=self,
        #     iuid=f"{hash(self)}:{self.iu_counter}",
        #     previous_iu=self._previous_iu,
        #     grounded_in=None,
        # )
        # output_iu.payload = message
        self.iu_counter += 1
        self._previous_iu = output_iu
        update_message = retico_core.UpdateMessage()

        if "update_type" not in frame.headers:
            print("Incoming IU has no update_type!")
            update_message.add_iu(output_iu, retico_core.UpdateType.ADD)
        elif frame.headers["update_type"] == "UpdateType.ADD":
            update_message.add_iu(output_iu, retico_core.UpdateType.ADD)
        elif frame.headers["update_type"] == "UpdateType.REVOKE":
            update_message.add_iu(output_iu, retico_core.UpdateType.REVOKE)
        elif frame.headers["update_type"] == "UpdateType.COMMIT":
            update_message.add_iu(output_iu, retico_core.UpdateType.COMMIT)


class AMQWriter(retico_core.AbstractModule):
    """
    Module providing a retico system with ActiveMQ message sending.
    The module will transform the AMQIU received into ActiveMQ messages, and send them to ActiveMQ following the AMQIU's `destination` and `headers` parameters.
    """

    @staticmethod
    def name():
        return "ActiveMQ Writer Module"

    @staticmethod
    def description():
        return "A Module providing a retico system with ActiveMQ message sending"

    @staticmethod
    def output_iu():
        return None

    @staticmethod
    def input_ius():
        return [AMQIU]

    def __init__(self, ip, port, **kwargs):
        """Initializes the ActiveMQWriter.

        Args:
            ip (str): the IP of the computer.
            port (str): the port corresponding to ActiveMQ

        """
        super().__init__(**kwargs)
        hosts = [(ip, port)]
        self.conn = stomp.Connection(host_and_ports=hosts, auto_content_length=False)
        self.conn.connect("admin", "admin", wait=True)

    def process_update(self, update_message):
        """
        This assumes that the message is json formatted, then packages it as payload into an IU
        """

        for amq_iu, ut in update_message:

            decorated_iu = amq_iu.get_deco_iu()

            # # if we want to send a RETICO IU type message
            # body = {}
            # body["payload"] = input_iu.get_deco_iu.__dict__["payload"]
            # body["update_type"] = str(ut)
            # print("dict = ", body)

            # if we just want to send the payload
            body = decorated_iu.payload

            # send the message to the correct destination
            # print(
            #     f"sent {body},  to : {amq_iu.destination} , with headers : {amq_iu.headers}"
            # )
            self.conn.send(
                body=body,
                destination=amq_iu.destination,
                headers=amq_iu.headers,
                persistent=True,
            )

        return None

    def process_update_2(self, update_message):
        """
        This assumes that the message is json formatted, then packages it as payload into an IU
        """

        for amq_iu, ut in update_message:

            # create a JSOn from all decorated IU extracted information
            decorated_iu = amq_iu.get_deco_iu()
            body = json.dumps(decorated_iu.__dict__)

            # send the message to the correct destination
            print(
                f"sent {body},  to : {amq_iu.destination} , with headers : {amq_iu.headers}"
            )
            self.conn.send(
                body=body,
                destination=amq_iu.destination,
                headers=amq_iu.headers,
                persistent=True,
            )

        return None


class TextAnswertoBEATBridge(retico_core.AbstractModule):

    @staticmethod
    def name():
        return "TextAnswertoBEATBridge Module"

    @staticmethod
    def description():
        return "A Module providing a way to tranform transcript outputted from ASR into BEAT input format"

    @staticmethod
    def output_iu():
        return AMQIU

    @staticmethod
    def input_ius():
        return [IncrementalUnit]

    def __init__(self, **kwargs):
        """Initializes the TextAnswertoBEATBridge.

        Args: topic(str): the topic/scope where the information will be read.

        """
        super().__init__(**kwargs)
        self.headers = [
            {"ELVISH_SCOPE": "DEFAULT_SCOPE", "MESSAGE_PREFIX": "vrExpress"},
            {"ELVISH_SCOPE": "DEFAULT_SCOPE", "MESSAGE_PREFIX": "vrSpeak"},
        ]
        self.destinations = ["/topic/DEFAULT_SCOPE", "/topic/vrSSML"]
        self.buffer = []
        self.iu_counter = 0

    # def create_iu(self, decorated_iu, destination, headers, grounded_iu=None):
    #     iu = super().create_iu(
    #         grounded_iu=grounded_iu,
    #         decorated_iu=decorated_iu,
    #         destination=destination,
    #         headers=headers,
    #     )
    #     # iu.set_amq(
    #     #     decorated_iu=decorated_iu,
    #     #     headers=headers,
    #     #     destination=destination,
    #     # )
    #     return iu

    def process_update(self, update_message):

        um = retico_core.abstract.UpdateMessage()

        final = False
        for input_iu, ut in update_message:
            if ut == retico_core.UpdateType.COMMIT:
                if input_iu.final:
                    final = True
                else:
                    self.buffer.append(input_iu.payload)

        if final:
            if len(self.buffer) == 1:
                complete_text = self.buffer[0]
            else:
                complete_text = "".join(self.buffer)
            print(complete_text)
            if complete_text is None:
                print("complete text is NONE :", [iu for iu in update_message])

            # combo_list = [
            #     ["greetings", "greeting", "NONE"],
            #     ["introduction", "introduce", "NONE"],
            #     ["farewell", "pre_closing", "NONE"],
            #     ["NONE", "NONE", "NONE"],
            # ]
            # combo = combo_list[random.randrange(len(combo_list))]

            # from SARA NLG module
            non_ascii = complete_text.replace("'", "")
            ssml_utterance, beat_utterance = non_ascii, non_ascii
            ssml_utterance = "<speak>" + ssml_utterance + "</speak>"
            none = "NONE"
            body = f"""vrExpress Brad user 1480100642410 <?xml version="1.0" encoding="UTF-8" standalone="no" ?> 
            <act> <participant id="Brad" role="actor" /><fml> 
            <turn continuation="false" />
            <sentence intention="{none}" strategy="{none}" rapport="{none}" text="{beat_utterance}" />
            </fml>
            <bml>
            <speech>"{beat_utterance}"</speech>
            </bml>
            <ssml>
            <speech><s>"{beat_utterance}"</s></speech>
            </ssml>
            </act>"""
            # create the decorated IU (cannot use classical create_iu from AbstractModule)
            try:
                beat_iu = retico_core.text.TextIU(
                    creator=self,
                    iuid=f"{hash(self)}:{self.iu_counter}",
                    previous_iu=self._previous_iu,
                    grounded_in=None,
                )
                self.iu_counter += 1
                self._previous_iu = output_iu
                beat_iu.payload = beat_utterance
                ssml_iu = retico_core.text.TextIU(
                    creator=self,
                    iuid=f"{hash(self)}:{self.iu_counter}",
                    previous_iu=self._previous_iu,
                    grounded_in=None,
                )
                self.iu_counter += 1
                self._previous_iu = output_iu
                ssml_iu.payload = ssml_utterance
                default_scope_iu = retico_core.text.TextIU(
                    creator=self,
                    iuid=f"{hash(self)}:{self.iu_counter}",
                    previous_iu=self._previous_iu,
                    grounded_in=None,
                )
                self.iu_counter += 1
                self._previous_iu = output_iu
                default_scope_iu.payload = body
            except Exception as e:
                log_exception(module=self, exception=e)

            # create AMQIU
            output_iu = self.create_iu(
                decorated_iu=ssml_iu,
                destination="/topic/vrSSML",
                headers=self.headers[1],
            )
            um.add_iu(output_iu, retico_core.UpdateType.COMMIT)
            output_iu = self.create_iu(
                decorated_iu=beat_iu,
                destination="/topic/vrNLG",
                headers=self.headers[1],
            )
            um.add_iu(output_iu, retico_core.UpdateType.COMMIT)
            output_iu = self.create_iu(
                decorated_iu=beat_iu,
                destination="/topic/vrNLGWOZ",
                headers=self.headers[1],
            )
            um.add_iu(output_iu, retico_core.UpdateType.COMMIT)
            output_iu = self.create_iu(
                decorated_iu=default_scope_iu,
                destination=self.destinations[0],
                headers=self.headers[0],
            )
            um.add_iu(output_iu, retico_core.UpdateType.COMMIT)

            self.buffer = []
            return um

    # def process_update(self, update_message):
    #     """
    #     This assumes that the message is json formatted, then packages it as payload into an IU
    #     """
    #     um = retico_core.abstract.UpdateMessage()

    #     msg = []
    #     for input_iu, ut in update_message:
    #         if ut == retico_core.UpdateType.COMMIT:
    #             if not input_iu.final:
    #                 msg.append(input_iu.payload)

    #     if len(msg) == 0:
    #         return None
    #     elif len(msg) == 1:
    #         complete_text = msg[0]
    #     else:
    #         complete_text = " ".join(msg)
    #     print(complete_text)
    #     if complete_text is None:
    #         print("complete text is NONE :", [iu for iu in update_message])

    #     # create BEAT format message from the IU's text
    #     ssml_string = ""
    #     for st in msg[:-1]:
    #         ssml_string += st + """ <break strength="medium"/>"""
    #     ssml_string += msg[-1]

    #     # phases = [
    #     #     "recommend_session",
    #     #     "find_session_detail",
    #     #     "find_person",
    #     #     "recommend_people",
    #     #     "like",
    #     #     "food",
    #     #     "dislike",
    #     #     "party",
    #     #     "introduction",
    #     #     "farewell",
    #     #     "reset",
    #     #     "greeting",
    #     #     "positive_confirmation",
    #     #     "negative_confirmation",
    #     #     "user_tells_about_work",
    #     #     "gratitude",
    #     #     "self_naming,",
    #     # ]
    #     # phase = phases[random.randrange(len(phases))]
    #     # phase = "greetings"

    #     # strategies = [
    #     #     "baseline",
    #     #     "None",
    #     #     "SD",
    #     #     "VSN",
    #     #     "QE",
    #     #     "ASN",
    #     #     "VSN",
    #     #     "BC",
    #     #     "PR",
    #     #     "RSE",
    #     # ]
    #     # strategy = strategies[random.randrange(len(strategies))]
    #     # strategy = "NONE"

    #     combo_not_working = [
    #         ["NONE", "NONE", "NONE"],
    #         [
    #             "session_recommendation",
    #             "indicate_start_of_session_recommendation",
    #             "NONE",
    #         ],
    #         [
    #             "goal_elicitation",
    #             "start_goal_elicitation",
    #             "NONE",
    #         ],
    #         [
    #             "food_recommendation",
    #             "indicate_start_of_food_recommendation_as_first_of_multiple_goals",
    #             "NONE",
    #         ],
    #     ]

    #     combo_list = [
    #         ["greetings", "greeting", "NONE"],
    #         ["introduction", "introduce", "NONE"],
    #         ["farewell", "pre_closing", "NONE"],
    #         ["NONE", "NONE", "NONE"],
    #     ]
    #     combo = combo_list[random.randrange(len(combo_list))]

    #     formatted_text = f"""vrExpress Brad user 1480100642410 <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
    #     <act> <participant id="Brad" role="actor" /><fml>
    #     <turn continuation="false" />
    #     <affect type="neutral" target="addressee">
    #     </affect> <culture type="neutral"> </culture>
    #     <personality type="neutral"> </personality>
    #     <sentence phase="{combo[0]}" intention="{combo[1]}" strategy="{combo[2]}" rapport="NONE" text="{complete_text}" />
    #     </fml>
    #     <bml>
    #     <speech>
    #     {complete_text}</speech>
    #     </bml>
    #     <ssml>
    #     <speech><s>{ssml_string}</s></speech>
    #     </ssml>
    #     </act>"""

    #     print("formatted_text = ", formatted_text)
    #     text_iu = retico_core.text.TextIU(iuid=0)
    #     text_iu.payload = formatted_text

    #     # create AMQIU
    #     output_iu = self.create_iu(decorated_iu=text_iu)
    #     um.add_iu(output_iu, retico_core.UpdateType.COMMIT)

    #     return um


class AMQWriterOpening(retico_core.AbstractModule):

    @staticmethod
    def name():
        return "ActiveMQ Writer Module"

    @staticmethod
    def description():
        return "A Module providing writing onto a ActiveMQ bus"

    @staticmethod
    def output_iu():
        return None

    @staticmethod
    def input_ius():
        return [retico_core.IncrementalUnit]

    def __init__(self, ip, port, **kwargs):
        """Initializes the ActiveMQWriter.

        Args: topic(str): the topic/scope where the information will be read.

        """
        super().__init__(**kwargs)
        hosts = [(ip, port)]
        self.conn = stomp.Connection(host_and_ports=hosts, auto_content_length=False)
        self.conn.connect("admin", "admin", wait=True)
        # self.socket.bind("tcp://*:5555")

    def process_update(self, update_message):
        """
        This assumes that the message is json formatted, then packages it as payload into an IU
        """
        for input_iu, um in update_message:

            d = {}
            d["payload"] = input_iu.__dict__["payload"]
            d["update_type"] = str(um)
            print("dict = ", d)

        headers = dict()
        headers["ELVISH_SCOPE"] = "DEFAULT_SCOPE"
        headers["MESSAGE_PREFIX"] = "vrWEFUI"

        body1 = 'vrSpeak Brad user 1000037 <?xml version="1.0" encoding="utf-16"?><act><participant id="brad" role="actor" /><bml><speech id="sp1" ref="voice_defaultTTS" type="application/ssml+xml"><mark name="T1" />hi <mark name="T2" /><mark name="T2" />i am <mark name="T3" /><mark name="T3" />sara <mark name="T4" /><mark name="T4" />what is <mark name="T5" /><mark name="T5" />your <mark name="T6" /><mark name="T6" />name <mark name="T7" /></speech><event message="vrAgentSpeech partial 1480100645859 T1 hi " stroke="sp1:T1" /><event message="vrAgentSpeech partial 1480100645859 T3 hi i am " stroke="sp1:T3" /><event message="vrAgentSpeech partial 1480100645859 T5 hi i am sara " stroke="sp1:T5" /><event message="vrAgentSpeech partial 1480100645859 T7 hi i am sara what is " stroke="sp1:T7" /><event message="vrAgentSpeech partial 1480100645859 T9 hi i am sara what is your " stroke="sp1:T9" /><event message="vrAgentSpeech partial 1480100645859 T11 hi i am sara what is your name " stroke="sp1:T11" /><sbm:event message="vrSpoke brad user 1480100642410 hi i am sara what is your name." stroke="sp1:relax" xmlns:xml="http://www.w3.org/XML/1998/namespace" xmlns:sbm="http://ict.usc.edu" /><intonation_break start="sp1:T0" end="sp1:T3"/><intonation_tone endtone="L-H%" start="sp1:T0" end="sp1:T4"/><intonation_tone endtone="L-L%" start="sp1:T3" end="sp1:T4"/><face type="FACS" au="2" amount="5.0"/><animation stroke="sp1:T3" priority="1" name="beat_low_right_Sara"/><head type="NOD" amount="0.1" repeats="1" relax="sp1:T3" priority="1" /><face type="FACS" stroke="sp1:T5" au="101" amount="1.0" priority="1" /><face type="FACS" start="sp1:T0" end="sp1:T0" au="103" amount="0.5"/><animation start="sp1:T0" priority="1" name="beat_middle_right_Sara"/><gaze participant = "Brad" target="Front" angle="0" start="7" sbm:joint-range="EYES" xmlns:sbm="http://ict.usc.edu"/></bml></act>'
        # body2 = 'vrSpeak Brad User 1466989728367 <?xml version="1.0" encoding="utf-16"?><act><participant id="Brad" role="actor" /><bml><event message="vrSpoke Brad User 6 " xmlns:sbm="​http://ict.usc.edu​" /> <animation priority="1" name="beat_low_right_sara" /></bml></act>'

        body3 = "vrWEFUI start"

        headers["MESSAGE_PREFIX"] = "vrExpress"
        self.conn.send(
            body=body3,
            destination="/topic/DEFAULT_SCOPE",
            headers=headers,
            persistent=True,
        )

        headers["MESSAGE_PREFIX"] = "vrSpeak"
        self.conn.send(
            body=body1,
            destination="/topic/DEFAULT_SCOPE",
            headers=headers,
            persistent=True,
        )

        return None


class fakeBEATSARA(retico_core.AbstractModule):
    """A Module simulating BEAT output to controll SARA body. Every time it receives an IU (from ASR), it sends a fixed BEAT output message, through ActiveMQ.

    Args:
        retico_core (_type_): _description_

    Returns:
        _type_: _description_
    """

    @staticmethod
    def name():
        return "fakeBEATSARA Module"

    @staticmethod
    def description():
        return "A Module simulating BEAT output to controll SARA body"

    @staticmethod
    def output_iu():
        return None

    @staticmethod
    def input_ius():
        return [retico_core.IncrementalUnit]

    def __init__(self, ip, port, **kwargs):
        """Initializes the ActiveMQWriter.

        Args: topic(str): the topic/scope where the information will be read.

        """
        super().__init__(**kwargs)
        self.queue = deque()
        hosts = [(ip, port)]
        self.conn = stomp.Connection(host_and_ports=hosts, auto_content_length=False)
        self.conn.connect("admin", "admin", wait=True)
        # self.socket.bind("tcp://*:5555")

    def process_update(self, update_message):
        """
        This assumes that the message is json formatted, then packages it as payload into an IU
        """

        # testing with a fixed headers and XML body from BEAT
        headers = dict()
        headers["ELVISH_SCOPE"] = "DEFAULT_SCOPE"

        # step 1
        headers["MESSAGE_PREFIX"] = "vrExpress"
        body = "vrWEFUI start"
        self.conn.send(
            body=body,
            destination="/topic/DEFAULT_SCOPE",
            headers=headers,
            persistent=True,
        )

        # step 2
        headers["MESSAGE_PREFIX"] = "vrSpeak"
        body = 'vrSpeak Brad user 1000037 <?xml version="1.0" encoding="utf-16"?><act><participant id="brad" role="actor" /><bml><speech id="sp1" ref="voice_defaultTTS" type="application/ssml+xml"><mark name="T1" />hi <mark name="T2" /><mark name="T2" />i am <mark name="T3" /><mark name="T3" />sara <mark name="T4" /><mark name="T4" />what is <mark name="T5" /><mark name="T5" />your <mark name="T6" /><mark name="T6" />name <mark name="T7" /></speech><event message="vrAgentSpeech partial 1480100645859 T1 hi " stroke="sp1:T1" /><event message="vrAgentSpeech partial 1480100645859 T3 hi i am " stroke="sp1:T3" /><event message="vrAgentSpeech partial 1480100645859 T5 hi i am sara " stroke="sp1:T5" /><event message="vrAgentSpeech partial 1480100645859 T7 hi i am sara what is " stroke="sp1:T7" /><event message="vrAgentSpeech partial 1480100645859 T9 hi i am sara what is your " stroke="sp1:T9" /><event message="vrAgentSpeech partial 1480100645859 T11 hi i am sara what is your name " stroke="sp1:T11" /><sbm:event message="vrSpoke brad user 1480100642410 hi i am sara what is your name." stroke="sp1:relax" xmlns:xml="http://www.w3.org/XML/1998/namespace" xmlns:sbm="http://ict.usc.edu" /><intonation_break start="sp1:T0" end="sp1:T3"/><intonation_tone endtone="L-H%" start="sp1:T0" end="sp1:T4"/><intonation_tone endtone="L-L%" start="sp1:T3" end="sp1:T4"/><face type="FACS" au="2" amount="5.0"/><animation stroke="sp1:T3" priority="1" name="beat_low_right_Sara"/><head type="NOD" amount="0.1" repeats="1" relax="sp1:T3" priority="1" /><face type="FACS" stroke="sp1:T5" au="101" amount="1.0" priority="1" /><face type="FACS" start="sp1:T0" end="sp1:T0" au="103" amount="0.5"/><animation start="sp1:T0" priority="1" name="beat_middle_right_Sara"/><gaze participant = "Brad" target="Front" angle="0" start="7" sbm:joint-range="EYES" xmlns:sbm="http://ict.usc.edu"/></bml></act>'
        self.conn.send(
            body=body,
            destination="/topic/DEFAULT_SCOPE",
            headers=headers,
            persistent=True,
        )

        # headers = dict()
        # headers["ELVISH_SCOPE"] = "DEFAULT_SCOPE"
        # headers["MESSAGE_PREFIX"] = "vrWEFUI"

        # body1 = 'vrSpeak Brad user 1000037 <?xml version="1.0" encoding="utf-16"?><act><participant id="brad" role="actor" /><bml><speech id="sp1" ref="voice_defaultTTS" type="application/ssml+xml"><mark name="T1" />hi <mark name="T2" /><mark name="T2" />i am <mark name="T3" /><mark name="T3" />sara <mark name="T4" /><mark name="T4" />what is <mark name="T5" /><mark name="T5" />your <mark name="T6" /><mark name="T6" />name <mark name="T7" /></speech><event message="vrAgentSpeech partial 1480100645859 T1 hi " stroke="sp1:T1" /><event message="vrAgentSpeech partial 1480100645859 T3 hi i am " stroke="sp1:T3" /><event message="vrAgentSpeech partial 1480100645859 T5 hi i am sara " stroke="sp1:T5" /><event message="vrAgentSpeech partial 1480100645859 T7 hi i am sara what is " stroke="sp1:T7" /><event message="vrAgentSpeech partial 1480100645859 T9 hi i am sara what is your " stroke="sp1:T9" /><event message="vrAgentSpeech partial 1480100645859 T11 hi i am sara what is your name " stroke="sp1:T11" /><sbm:event message="vrSpoke brad user 1480100642410 hi i am sara what is your name." stroke="sp1:relax" xmlns:xml="http://www.w3.org/XML/1998/namespace" xmlns:sbm="http://ict.usc.edu" /><intonation_break start="sp1:T0" end="sp1:T3"/><intonation_tone endtone="L-H%" start="sp1:T0" end="sp1:T4"/><intonation_tone endtone="L-L%" start="sp1:T3" end="sp1:T4"/><face type="FACS" au="2" amount="5.0"/><animation stroke="sp1:T3" priority="1" name="beat_low_right_Sara"/><head type="NOD" amount="0.1" repeats="1" relax="sp1:T3" priority="1" /><face type="FACS" stroke="sp1:T5" au="101" amount="1.0" priority="1" /><face type="FACS" start="sp1:T0" end="sp1:T0" au="103" amount="0.5"/><animation start="sp1:T0" priority="1" name="beat_middle_right_Sara"/><gaze participant = "Brad" target="Front" angle="0" start="7" sbm:joint-range="EYES" xmlns:sbm="http://ict.usc.edu"/></bml></act>'
        # # body2 = 'vrSpeak Brad User 1466989728367 <?xml version="1.0" encoding="utf-16"?><act><participant id="Brad" role="actor" /><bml><event message="vrSpoke Brad User 6 " xmlns:sbm="​http://ict.usc.edu​" /> <animation priority="1" name="beat_low_right_sara" /></bml></act>'

        # body3 = "vrWEFUI start"

        # headers["MESSAGE_PREFIX"] = "vrExpress"
        # self.conn.send(
        #     body=body3,
        #     destination="/topic/DEFAULT_SCOPE",
        #     headers=headers,
        #     persistent=True,
        # )

        # headers["MESSAGE_PREFIX"] = "vrSpeak"
        # self.conn.send(
        #     body=body1,
        #     destination="/topic/DEFAULT_SCOPE",
        #     headers=headers,
        #     persistent=True,
        # )

        return None


class fakeTTSSARA(AbstractProducingModule):

    @staticmethod
    def name():
        return "fakeTTSSARA Module"

    @staticmethod
    def description():
        return "A Module faking to be SARA's TTS Module"

    @staticmethod
    def output_iu():
        return retico_core.IncrementalUnit

    @staticmethod
    def input_ius():
        return None

    def __init__(self, speech_file_path, mark_file_path, ip, port, **kwargs):
        """Initializes the fakeTTSSARA Module.

        Args: topic(str): the topic/scope where the information will be read.

        """
        super().__init__(**kwargs)

        absolute_path = os.path.dirname(__file__)
        self.speech_file_path = os.path.join(absolute_path, speech_file_path)
        self.mark_file_path = os.path.join(absolute_path, mark_file_path)

        # self.conn.subscribe(destination="/topic/vrSSML", id=1, ack="auto")
        # self.conn.subscribe(destination="/topic/DEFAULT_SCOPE", id=1, ack="auto")

        destination = "/topic/DEFAULT_SCOPE"

        hosts = [(ip, port)]
        self.conn = stomp.Connection(host_and_ports=hosts, auto_content_length=False)
        self.conn.connect("admin", "admin", wait=True)
        self.conn.set_listener("", self.Listener(self))
        self.conn.subscribe(destination=destination, id=1, ack="auto")
        # self.socket.bind("tcp://*:5555")
        self.destination = destination
        # self.frame_buffer = []

    class Listener(stomp.ConnectionListener):
        def __init__(self, module):
            super().__init__()
            # in order to use methods of activeMQ we create its instance
            self.module = module

        # Override the methods on_error and on_message provides by the parent class
        def on_error(self, frame):
            self.module.on_listener_error(frame)
            # print('received an error "%s"' % frame.body)

        def on_message(self, frame):
            # self.module.logMessageReception(frame)
            self.module.on_message(frame)

    def on_message(self, frame):
        message = frame.body
        destination = frame.headers["destination"]

        # print("TTS message, destination = ", destination)

        if destination == "/topic/vrSSML":
            pass
        #     speech_stream = self.synthesis(text=message)
        #     mark_stream = self.generateMark(message)
        #     self.storeFile(speech_stream, type="speech")
        #     self.storeFile(mark_stream, type="mark")

        elif destination == "/topic/DEFAULT_SCOPE":

            message = message.split(" ")
            # print("message = ", message)
            if "RemoteSpeechCmd" == message[0]:
                # print("REMOTE")
                self.sentence_id = message[1].split("+")[2]

                # cmd = " ".join(message[1:])
                # idx = cmd.index('<')
                # text = cmd[idx:]
                # text = self.removeXMLTags(text)
                # text = self.removePunctuationMarks(text)

                with open(self.mark_file_path) as f:
                    mark_json = json.load(f)

                if mark_json != {}:
                    reply = self.generateMessage(mark_json)

                    headers = dict()
                    headers["ELVISH_SCOPE"] = "DEFAULT_SCOPE"
                    headers["MESSAGE_PREFIX"] = "RemoteSpeechReply"
                    self.send_message(
                        message=reply, destination="DEFAULT_SCOPE", headers=headers
                    )

            elif "PlaySound" == message[0]:
                # print("playsound")
                # self.logger.debug("playsound")
                # self.play()
                pass

            elif "vrSpeak" == message[0]:
                text = unquote(message[1])
                text2 = text.replace("+", " ")
                print(f"vrSpeak message = {text2}\n")

    def send_message(
        self, message, destination, headers, isQueue=False, persistent=True
    ):
        """intialize the class

        Args:
            message (str): message to be sent
            destination (str): name of channel to be used for communication
            isQueue (bool): decide if channel is a queue or a topic (default : False)
        """

        if isQueue:
            self.conn.send(
                body=message,
                destination="/queue/" + destination,
                headers=headers,
                persistent=persistent,
            )
        else:
            self.conn.send(
                body=message,
                destination="/topic/" + destination,
                headers=headers,
                persistent=persistent,
            )

    # def synthesis(
    #     self,
    #     text,
    #     voiceId="Joanna",
    #     textType="ssml",
    #     outputFormat="mp3",
    #     engine="neural",
    # ):
    #     try:
    #         response = self.polly.synthesize_speech(
    #             Text=text,
    #             VoiceId=voiceId,
    #             Engine=engine,
    #             TextType=textType,
    #             OutputFormat=outputFormat,
    #         )

    #     except (BotoCoreError, ClientError) as error:
    #         # The service returned an error, exit gracefully
    #         # print(error)
    #         self.logger.error(error)
    #         sys.exit(-1)

    #     return response["AudioStream"]

    # def generateMark(
    #     self,
    #     text,
    #     voiceId="Joanna",
    #     textType="ssml",
    #     outputFormat="json",
    #     engine="neural",
    # ):
    #     speech_mark_types = ["sentence", "word", "viseme"]
    #     try:
    #         response = self.polly.synthesize_speech(
    #             Text=text,
    #             VoiceId=voiceId,
    #             Engine=engine,
    #             TextType=textType,
    #             OutputFormat=outputFormat,
    #             SpeechMarkTypes=speech_mark_types,
    #         )

    #     except (BotoCoreError, ClientError) as error:
    #         # The service returned an error, exit gracefully
    #         # print(error)
    #         self.logger.error(error)
    #         sys.exit(-1)

    #     return response["AudioStream"]

    # def storeFile(self, inputStream, type):
    #     if type == "speech":
    #         with closing(inputStream) as stream:
    #             try:
    #                 # Open a file for writing the output as a binary stream
    #                 with open(self.speech_file_path, "wb") as file:
    #                     file.write(stream.read())
    #             except IOError as error:
    #                 # Could not write to file, exit gracefully
    #                 # print(error)
    #                 self.logger.error(error)
    #                 sys.exit(-1)
    #     else:
    #         with closing(inputStream) as stream:
    #             try:
    #                 byte_string = stream.read()
    #                 byte_string_list = (
    #                     byte_string.decode("utf8").replace("'", '"').split("\n")
    #                 )
    #                 data = []
    #                 for b in byte_string_list[:-1]:
    #                     data.append(literal_eval(b))

    #                 with open(self.mark_file_path, "w") as f:
    #                     json.dump(data, f)

    #             except IOError as error:
    #                 # Could not write to file, exit gracefully
    #                 # print(error)
    #                 self.logger.error(error)
    #                 sys.exit(-1)

    def play(self):
        # print(sys.path)
        # print(os.path)
        # cwd = Path.cwd()
        # print(cwd)
        os.startfile(self.speech_file_path)
        # if sys.platform == "win32":
        #     os.startfile(self.speech_file_path)
        # else:
        #     # The following works on macOS and Linux. (Darwin = mac, xdg-open = linux)
        #     opener = "open" if sys.platform == "darwin" else "xdg-open"
        #     subprocess.call([opener, self.speech_file_path])

    def removeXMLTags(self, text):
        for char in "<[^>]+>":
            text = text.replace(char, "")

        text = text.replace("   ", " ")
        text = text.replace("  ", " ")
        text = text.replace("\n", "")
        text = text.strip()

        return text

    def removePunctuationMarks(self, text):
        text = text.replace("\\", "")
        text = text.replace("   ", " ")
        text = text.replace("  ", " ")
        text = text.replace(",", "")
        text = text.replace('"', "")

        return text

    def convertVisemes(self, input):

        output = "open"
        if input == "f":
            output = "FV"
        elif input == "i":
            output = "wide"
        elif input == "k":
            output = "tRoof"
        elif input == "p":
            output = "PBM"
        elif input == "r":
            output = "tBack"
        elif input == "S":
            output = "ShCh"
        elif input == "s":
            output = "ShCh"
        elif input == "T":
            output = "tTeeth"
        elif input == "t":
            output = "tTeeth"
        elif input == "@":
            output = "open"
        elif input == "a":
            output = "open"
        elif input == "e":
            output = "tBack"
        elif input == "E":
            output = "tBack"
        elif input == "i":
            output = "wide"
        elif input == "o":
            output = "open"
        elif input == "O":
            output = "open"
        elif input == "u":
            output = "W"

        return output

    def generateMessage(self, marks):
        message = (
            "RemoteSpeechReply Brad "
            + str(self.sentence_id)
            + ' OK: <?xml version="1.0" encoding="UTF-8"?>\n<speech>\n'
        )
        message += '<soundFile name="' + self.speech_file_path + '" />\n'
        openWord = False
        markTime = 2
        endTimes = []
        finalTime = 0

        for idx, mark in enumerate(marks):
            if mark["type"] == "word":
                if openWord:
                    message += (
                        '<mark name="T'
                        + str(markTime)
                        + '" time="*ENDTIME_'
                        + str(markTime - 1)
                        + '*"/>\n'
                        + "</word>\n\n"
                    )

                message += (
                    '<mark name="T'
                    + str(markTime)
                    + '" time="'
                    + str(mark["time"] / 1000)
                    + '"/>\n'
                )
                openWord = True
                endTimes.append(str(mark["time"] / 1000))

                message += (
                    '<word start="'
                    + str(mark["time"] / 1000)
                    + '" end="*ENDTIME_'
                    + str(markTime)
                    + "*"
                    + '">\n'
                )
                markTime += 1

            elif mark["type"] == "viseme":
                # print(self.convertVisemes(mark['value']))
                # self.logger.debug("print", visemes=self.convertVisemes(mark["value"]))
                # too many visemes make the lip look bad, so just take 3 out of 5 visemes
                if idx % 5 != 2 and idx % 5 != 4:
                    message += (
                        '<viseme start="'
                        + str(mark["time"] / 1000)
                        + '" articulation="0.3" type="'
                        + self.convertVisemes(mark["value"])
                        + '" />\n'
                    )
                finalTime = mark["time"] / 1000

        message += (
            '<mark name="T'
            + str(markTime)
            + '" time="*ENDTIME_'
            + str(markTime)
            + '*"/>\n'
        )
        message += "</word>\n"
        message += "</speech>"

        for i in range(len(endTimes)):
            message = message.replace("*ENDTIME_" + str(i) + "*", endTimes[i])

        message = message.replace(
            "*ENDTIME_" + str(len(endTimes)) + "*", str(finalTime)
        )
        message = message.replace("*ENDTIME_" + str(markTime) + "*", str(finalTime))

        return message
