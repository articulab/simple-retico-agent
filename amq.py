"""
ZeroMQ Module
=============

This module defines two incremental modules ZeroMQReader and ZeroMQWriter that act as a
a bridge between ZeroMQ and retico. For this, a ZeroMQIU is defined that contains the
information revceived over the ZeroMQ bridge.
"""

# retico
import os
from pathlib import Path
import subprocess
import sys
import retico_core
from retico_core.abstract import *

# zeromq & supporting libraries
import zmq, json
import threading
import datetime
import time
from collections import deque

import stomp
import json


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
        self.queue = deque()
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


class AMQIU(retico_core.IncrementalUnit):

    @staticmethod
    def type():
        return "AMQ IU"

    def get_payload(self):
        """Return the text contained in the IU.

        Returns:
            str: The text contained in the IU.
        """
        return self.payload

    def set_payload(self, payload):
        """Sets the text contained in the IU.

        Args:
            text (str): The new text of the IU
        """
        self.payload = payload


class AMQWriter(retico_core.AbstractModule):

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

    def __init__(self, ip, port, headers, destination, **kwargs):
        """Initializes the ActiveMQWriter.

        Args: topic(str): the topic/scope where the information will be read.

        """
        super().__init__(**kwargs)
        self.queue = deque()
        hosts = [(ip, port)]
        self.conn = stomp.Connection(host_and_ports=hosts, auto_content_length=False)
        self.conn.connect("admin", "admin", wait=True)
        # self.socket.bind("tcp://*:5555")
        self.headers = headers
        self.destination = destination

    def process_update(self, update_message):
        """
        This assumes that the message is json formatted, then packages it as payload into an IU
        """
        # print("ZMQ Writer process update", self.topic)

        # assert self.headers == ["ELVISH_SCOPE", "MESSAGE_PREFIX"]

        for input_iu, um in update_message:

            # if the message body is created from IU's data
            d = {}
            d["payload"] = input_iu.__dict__["payload"]
            d["update_type"] = str(um)
            print("dict = ", d)
            body = d
            headers = self.headers

            # testing with a fixed headers and XML body from BEAT
            body = "vrWEFUI start"
            headers = dict()
            headers["ELVISH_SCOPE"] = "DEFAULT_SCOPE"
            headers["MESSAGE_PREFIX"] = "vrExpress"

            # # send the message to the correct destination
            # self.conn.send(
            #     body=body,
            #     destination=self.destination,
            #     headers=self.headers,
            #     persistent=True,
            # )

            # step 1
            headers["MESSAGE_PREFIX"] = "vrExpress"
            body3 = "vrWEFUI start"
            self.conn.send(
                body=body3,
                destination="/topic/DEFAULT_SCOPE",
                headers=headers,
                persistent=True,
            )

            # step 2
            headers["MESSAGE_PREFIX"] = "vrSpeak"
            body1 = 'vrSpeak Brad user 1000037 <?xml version="1.0" encoding="utf-16"?><act><participant id="brad" role="actor" /><bml><speech id="sp1" ref="voice_defaultTTS" type="application/ssml+xml"><mark name="T1" />hi <mark name="T2" /><mark name="T2" />i am <mark name="T3" /><mark name="T3" />sara <mark name="T4" /><mark name="T4" />what is <mark name="T5" /><mark name="T5" />your <mark name="T6" /><mark name="T6" />name <mark name="T7" /></speech><event message="vrAgentSpeech partial 1480100645859 T1 hi " stroke="sp1:T1" /><event message="vrAgentSpeech partial 1480100645859 T3 hi i am " stroke="sp1:T3" /><event message="vrAgentSpeech partial 1480100645859 T5 hi i am sara " stroke="sp1:T5" /><event message="vrAgentSpeech partial 1480100645859 T7 hi i am sara what is " stroke="sp1:T7" /><event message="vrAgentSpeech partial 1480100645859 T9 hi i am sara what is your " stroke="sp1:T9" /><event message="vrAgentSpeech partial 1480100645859 T11 hi i am sara what is your name " stroke="sp1:T11" /><sbm:event message="vrSpoke brad user 1480100642410 hi i am sara what is your name." stroke="sp1:relax" xmlns:xml="http://www.w3.org/XML/1998/namespace" xmlns:sbm="http://ict.usc.edu" /><intonation_break start="sp1:T0" end="sp1:T3"/><intonation_tone endtone="L-H%" start="sp1:T0" end="sp1:T4"/><intonation_tone endtone="L-L%" start="sp1:T3" end="sp1:T4"/><face type="FACS" au="2" amount="5.0"/><animation stroke="sp1:T3" priority="1" name="beat_low_right_Sara"/><head type="NOD" amount="0.1" repeats="1" relax="sp1:T3" priority="1" /><face type="FACS" stroke="sp1:T5" au="101" amount="1.0" priority="1" /><face type="FACS" start="sp1:T0" end="sp1:T0" au="103" amount="0.5"/><animation start="sp1:T0" priority="1" name="beat_middle_right_Sara"/><gaze participant = "Brad" target="Front" angle="0" start="7" sbm:joint-range="EYES" xmlns:sbm="http://ict.usc.edu"/></bml></act>'
            self.conn.send(
                body=body1,
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


class AMQReader(retico_core.AbstractProducingModule):

    @staticmethod
    def name():
        return "ActiveMQ Reader Module"

    @staticmethod
    def description():
        return "A Module providing reading onto a ActiveMQ bus"

    @staticmethod
    def output_iu():
        return AMQIU

    def __init__(self, ip, port, destination, **kwargs):
        """Initializes the ActiveMQReader.

        Args: topic(str): the topic/scope where the information will be read.

        """
        super().__init__(**kwargs)
        self.queue = deque()
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
        # self.frame_buffer.append(frame)

        output_iu = self.create_iu()
        output_iu.set_payload(frame)
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

        # print(update_message)
        print([iu for (iu, ut) in update_message])
        self.append(update_message)
        # for queue in self._right_buffers:
        #     while not queue.empty():
        #         um = queue.get_nowait()
        #         print([iu for (iu, ut) in um])
        # print([[um for um in queue] for queue in self._right_buffers])
        # return update_message


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

        print("TTS message, destination = ", destination)

        if destination == "/topic/vrSSML":
            pass
        #     speech_stream = self.synthesis(text=message)
        #     mark_stream = self.generateMark(message)
        #     self.storeFile(speech_stream, type="speech")
        #     self.storeFile(mark_stream, type="mark")

        elif destination == "/topic/DEFAULT_SCOPE":
            message = message.split(" ")
            print("message = ", message)
            if "RemoteSpeechCmd" == message[0]:
                print("REMOTE")
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
                print("playsound")
                # self.logger.debug("playsound")
                self.play()

    # def process_update(self, update_message):

    #     print("TTS process update")

    #     for iu, ut in update_message:
    #         frame = iu.payload

    #         message = frame.body
    #         destination = frame.headers["destination"]

    #         print("TTS message, destination = ", destination)

    #         if destination == "/topic/vrSSML":
    #             pass
    #         #     speech_stream = self.synthesis(text=message)
    #         #     mark_stream = self.generateMark(message)
    #         #     self.storeFile(speech_stream, type="speech")
    #         #     self.storeFile(mark_stream, type="mark")

    #         elif destination == "/topic/DEFAULT_SCOPE":
    #             message = message.split(" ")
    #             if "RemoteSpeechCmd" == message[0]:
    #                 self.sentence_id = message[1].split("+")[2]

    #                 # cmd = " ".join(message[1:])
    #                 # idx = cmd.index('<')
    #                 # text = cmd[idx:]
    #                 # text = self.removeXMLTags(text)
    #                 # text = self.removePunctuationMarks(text)

    #                 with open(self.mark_file_path) as f:
    #                     mark_json = json.load(f)

    #                 if mark_json != {}:
    #                     reply = self.generateMessage(mark_json)

    #                     headers = dict()
    #                     headers["ELVISH_SCOPE"] = "DEFAULT_SCOPE"
    #                     headers["MESSAGE_PREFIX"] = "RemoteSpeechReply"
    #                     self.send_message(
    #                         message=reply, destination="DEFAULT_SCOPE", headers=headers
    #                     )

    #             elif "PlaySound" == message[0]:
    #                 # print('playsound')
    #                 # self.logger.debug("playsound")
    #                 self.play()

    #     return None

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
