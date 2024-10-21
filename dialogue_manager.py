"""
VAD Module
==================

A retico module using webrtcvad's Voice Activity Detection (VAD) to enhance AudioIUs with
turn-taking informations (like user turn, silence or interruption).
It takes AudioIUs as input and transform them into VADTurnAudioIUs by adding to it turn-taking
informations through the IU parameter vad_state.
It also takes TextAlignedAudioIUs as input (from the SpeakerModule), which provides information
on when the speakers are outputting audio (when the agent is talking).

The module considers that the current dialogue state (self.user_turn_text) can either be :
- the user turn
- the agent turn
- a silence between two turns

The transitions between the 3 dialogue states are defined as following :
- If, while the dialogue state is a silence and the received AudioIUS are recognized as
containing speech (VA = True), it considers that dialogue state switches to user turn, and sends
(ADD) these IUs with vad_state = "user_turn".
- If, while the dialogue state is user turn and a long silence is recognized (with a defined
threshold), it considers that it is a user end of turn (EOT). It then COMMITS all IUs
corresponding to current user turn (with vad_state = "user_turn") and dialogue state switches to
agent turn.
- If, while the dialogue state is agent turn, it receives the information that the SpeakerModule
has outputted the whole agent turn (a TextAlignedAudioIU with final=True), it considers that it
is an agent end of turn, and dialogue state switches to silence.
- If, while the dialogue state is agent turn and before receiving an agent EOT from
SpeakerModule, it recognize audio containing speech, it considers the current agent turn is
interrupted by the user (user barge-in), and sends this information to the other modules to make
the agent stop talking (by sending an empty IU with vad_state = "interruption"). Dialogue state
then switches to user turn.

Inputs : AudioIU, TextAlignedAudioIU

Outputs : VADTurnAudioIU
"""

from functools import partial
import json
import time
import pydub
import webrtcvad

import retico_core
from retico_core.audio import AudioIU
from additional_IUs import (
    VADTurnAudioIU,
    DMIU,
    VADIU,
    SpeakerAlignementIU,
)
from whisper_asr_interruption import SpeechRecognitionTurnIU


class DialogueHistory:

    def __init__(
        self,
        prompt_format_config_file,
        terminal_logger=None,
        file_logger=None,
        initial_system_prompt="",
        context_size=2000,
    ):
        self.terminal_logger = terminal_logger
        self.file_logger = file_logger
        with open(prompt_format_config_file, "r", encoding="utf-8") as config:
            self.prompt_format_config = json.load(config)
        self.initial_system_prompt = initial_system_prompt
        self.current_system_prompt = initial_system_prompt
        self.dialogue_history = [
            {
                "turn_id": -1,
                "speaker": "system_prompt",
                "text": initial_system_prompt,
            }
        ]
        self.cpt_0 = 1
        self.context_size = context_size

    # Formatters

    def format(self, config_id, text):
        return (
            self.prompt_format_config[config_id]["pre"]
            + self.format_role(config_id)
            + text
            + self.prompt_format_config[config_id]["suf"]
        )

    def format_system_prompt(self, system_prompt):
        return self.format(config_id="system_prompt", text=system_prompt)

    def format_sentence(self, utterance):
        return self.format(config_id=utterance["speaker"], text=utterance["text"])

    def format_role(self, config_id):
        if "role" in self.prompt_format_config[config_id]:
            return (
                self.prompt_format_config[config_id]["role"]
                + " "
                + self.prompt_format_config[config_id]["role_sep"]
                + " "
            )
        else:
            return ""

    # Setters

    def append_utterance(self, utterance):
        assert set(("turn_id", "speaker", "text")) <= set(utterance)
        self.dialogue_history.append(utterance)

    def reset_system_prompt(self):
        self.change_system_prompt(self.initial_system_prompt)

    def change_system_prompt(self, system_prompt):
        previous_system_prompt = self.current_system_prompt
        self.current_system_prompt = system_prompt
        self.dialogue_history[0]["text"] = system_prompt
        return previous_system_prompt

    def prepare_dialogue_history(self, fun_tokenize):
        """Calculate if the current dialogue history is bigger than the size threshold (short_memory_context_size).
        If the dialogue history contains too many tokens, remove the older dialogue turns until its size is smaller than the threshold.
        """

        prompt = self.get_prompt_cpt(self.cpt_0)
        prompt_tokens = fun_tokenize(bytes(prompt, "utf-8"))
        nb_tokens = len(prompt_tokens)
        while nb_tokens > self.context_size:
            self.cpt_0 += 1
            prompt = self.get_prompt_cpt(self.cpt_0)
            prompt_tokens = fun_tokenize(bytes(prompt, "utf-8"))
            nb_tokens = len(prompt_tokens)
        return prompt, prompt_tokens

    def interruption_alignment_new_agent_sentence(
        self, utterance, punctuation_ids, interrupted_speaker_iu
    ):
        """After an interruption, this function will align the sentence stored in dialogue history with the last word spoken by the agent.

        This function is triggered if the interrupted speaker IU has been received before the module has stored the new agent sentence in the dialogue history.
        If that is not the case, the function interruption_alignment_last_agent_sentence is triggered instead.

        With the informations stored in self.interrupted_speaker_iu, this function will shorten the new_agent_sentence to be aligned with the last words spoken by the agent.

        Args:
            new_agent_sentence (string): the utterance generated by the LLM, that has been interrupted by the user and needs to be aligned.
        """
        new_agent_sentence = self.format_sentence(utterance)
        self.terminal_logger.info(
            f"interruption alignement func {utterance} {new_agent_sentence}",
            debug=True,
        )
        # remove role
        new_agent_sentence = new_agent_sentence[
            len(self.format_role("agent")) : len(new_agent_sentence)
        ]

        # split the sentence into clauses
        sentence_clauses = []
        old_i = 0
        for i, c in enumerate(new_agent_sentence):
            if c in punctuation_ids or i == len(new_agent_sentence) - 1:
                sentence_clauses.append(new_agent_sentence[old_i : i + 1])
                old_i = i + 1

        # remove all clauses after clause_id (the interrupted clause)
        sentence_clauses = sentence_clauses[: interrupted_speaker_iu.clause_id + 1]

        # Shorten the last agent utterance until the last char outputted by the speakermodule before the interruption
        sentence_clauses[-1] = sentence_clauses[-1][
            : interrupted_speaker_iu.char_id + 1
        ]

        # Merge the clauses back together
        new_agent_sentence = "".join(sentence_clauses)

        # Add interruption suf
        new_agent_sentence += self.prompt_format_config["interruption"]["suf"]

        # format the sentence again with prefix, role and suffix
        new_agent_sentence = self.format("agent", new_agent_sentence)

        print("INTERRUPTED AGENT SENTENCE : ", new_agent_sentence.decode("utf-8"))

    # Getters

    def get_dialogue_history(self):
        return self.dialogue_history

    def get_prompt(self):
        prompt = ""
        for utterance in self.dialogue_history:
            prompt += self.format_sentence(utterance)
        return self.format("prompt", prompt)

    def get_prompt_cpt(self, start=1, end=None):
        if end is None:
            end = len(self.dialogue_history)
        assert start > 0
        assert end >= start
        prompt = self.format_system_prompt(self.dialogue_history[0]["text"])
        for utterance in self.dialogue_history[start:end]:
            prompt += self.format_sentence(utterance)
        return self.format("prompt", prompt)

    def get_prompt_with_specific_system_prompt(self, system_prompt):
        prompt = self.format_system_prompt(system_prompt)
        for utterance in self.dialogue_history[1:]:
            prompt += self.format_sentence(utterance)
        return self.format("prompt", prompt)

    def get_stop_patterns(self):
        c = self.prompt_format_config
        user_stop_pat = (
            c["user"]["role"] + " " + c["user"]["role_sep"],
            c["user"]["role"] + "" + c["user"]["role_sep"],
        )
        agent_stop_pat = (
            c["agent"]["role"] + " " + c["agent"]["role_sep"],
            c["agent"]["role"] + "" + c["agent"]["role_sep"],
        )
        return (user_stop_pat, agent_stop_pat)


class VADModule(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "VAD Module"

    @staticmethod
    def description():
        return "a module enhancing AudioIUs with VA activation for both user (using webrtcvad's VAD) and agent (using IUs received from Speaker Module)."

    @staticmethod
    def input_ius():
        return [AudioIU, SpeakerAlignementIU]

    @staticmethod
    def output_iu():
        return VADIU

    def __init__(
        self,
        printing=False,
        target_framerate=16000,
        input_framerate=44100,
        channels=1,
        sample_width=2,
        vad_aggressiveness=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.printing = printing
        self.target_framerate = target_framerate
        self.input_framerate = input_framerate
        self.channels = channels
        self.sample_width = sample_width
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.VA_agent = False

    def resample_audio(self, audio):
        """Resample the audio's frame_rate to correspond to self.target_framerate.

        Args:
            audio (bytes): the audio received from the microphone that could need resampling.

        Returns:
            bytes: the resampled audio chunk.
        """
        if self.input_framerate != self.target_framerate:
            s = pydub.AudioSegment(
                audio,
                sample_width=self.sample_width,
                channels=self.channels,
                frame_rate=self.input_framerate,
            )
            s = s.set_frame_rate(self.target_framerate)
            return s._data
        return audio

    def process_update(self, update_message):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L402

        Args:
            update_message (UpdateType): UpdateMessage that contains new IUs, if the IUs are ADD,
            they are added to the audio_buffer.
        """
        for iu, ut in update_message:
            # IUs from SpeakerModule, can be either agent BOT or EOT
            if isinstance(iu, SpeakerAlignementIU):
                if ut == retico_core.UpdateType.ADD:
                    # agent EOT
                    if iu.event == "agent_EOT":
                        self.VA_agent = False
                    if iu.event == "interruption":
                        self.VA_agent = False
                    # agent BOT
                    elif iu.event == "agent_BOT":
                        self.VA_agent = True
            elif isinstance(iu, AudioIU):
                if ut == retico_core.UpdateType.ADD:
                    if self.input_framerate != iu.rate:
                        raise ValueError(
                            f"input framerate differs from iu framerate : {self.input_framerate} vs {iu.rate}"
                        )
                    audio = self.resample_audio(iu.raw_audio)
                    VA_user = self.vad.is_speech(audio, self.target_framerate)
                    output_iu = self.create_iu(
                        grounded_in=iu,
                        audio=audio,
                        nframes=iu.nframes,
                        rate=self.input_framerate,
                        sample_width=self.sample_width,
                        va_user=VA_user,
                        va_agent=self.VA_agent,
                    )
                    um = retico_core.UpdateMessage.from_iu(
                        output_iu, retico_core.UpdateType.ADD
                    )
                    self.append(um)


class DialogueManagerModule(retico_core.AbstractModule):
    """

    Inputs : VADIU

    Outputs : DMIU
    """

    @staticmethod
    def name():
        return "DialogueManager Module"

    @staticmethod
    def description():
        return "a module that manage the dialogue"

    @staticmethod
    def input_ius():
        return [VADIU, SpeakerAlignementIU]

    @staticmethod
    def output_iu():
        return retico_core.IncrementalUnit

    def __init__(
        self,
        dialogue_history: DialogueHistory,
        printing=False,
        silence_dur=1,
        bot_dur=0.4,
        silence_threshold=0.75,
        input_framerate=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.printing = printing
        self.input_framerate = input_framerate
        self.channels = None
        self.sample_width = None
        self.nframes = None
        self.silence_dur = silence_dur
        self.bot_dur = bot_dur
        self.silence_threshold = silence_threshold
        self._n_sil_audio_chunks = None
        self._n_bot_audio_chunks = None
        self.buffer_pointer = 0
        self.dialogue_state = "opening"
        self.turn_id = 0
        self.repeat_timer = float("inf")
        self.overlap_timer = -float("inf")
        self.turn_beginning_timer = -float("inf")
        self.audio_ius_agent_last_turn = []
        self.dialogue_history = dialogue_history

        self.policies = {
            "user_speaking": {
                "silence_after_user": [
                    self.dialogue_history.reset_system_prompt,
                    partial(self.send_action, action="start_answer_generation"),
                ],
                "agent_overlaps_user": [
                    partial(self.send_action, "system_interruption"),
                    # self.check_turn_beginning_timer,
                ],
            },
            "agent_speaking": {
                "user_overlaps_agent": [
                    partial(self.send_action, "system_interruption"),
                    self.increment_turn_id,
                    # self.check_turn_beginning_timer,
                ],
                "silence_after_agent": [
                    partial(self.set_repeat_timer, 5),
                ],
            },
            "silence_after_agent": {
                "silence_after_agent": [
                    self.check_repeat_timer_2,
                ],
                "mutual_overlap": [
                    partial(self.send_action, "system_interruption"),
                ],
                "user_speaking": [
                    self.increment_turn_id,
                    # self.set_turn_beginning_timer,
                ],
            },
            "silence_after_user": {
                "mutual_overlap": [
                    partial(self.send_action, "system_interruption"),
                ],
                "user_speaking": [
                    self.increment_turn_id,
                    # self.set_turn_beginning_timer,
                ],
            },
            "mutual_overlap": {
                "silence_after_agent": [partial(self.reset_repeat_timer, 5)]
            },
            "agent_overlaps_user": {
                "silence_after_user": [
                    partial(self.reset_repeat_timer, 5),
                ],
            },
            "user_overlaps_agent": {
                "silence_after_agent": [
                    partial(self.reset_repeat_timer, 5),
                ],
            },
        }

    def get_n_audio_chunks(self, param_name, duration):
        """Returns the number of audio chunks containing speech needed in the audio buffer to have a BOT (beginning of turn)
        (ie. to how many audio_chunk correspond self.bot_dur)

        Returns:
            int: the number of audio chunks corresponding to the duration of self.bot_dur.
        """
        if not getattr(self, param_name):
            if len(self.current_input) == 0:
                return None
            first_iu = self.current_input[0]
            self.input_framerate = first_iu.rate
            self.nframes = first_iu.nframes
            self.sample_width = first_iu.sample_width
            # nb frames in each audio chunk
            nb_frames_chunk = len(first_iu.payload) / 2
            # duration of 1 audio chunk
            duration_chunk = nb_frames_chunk / self.input_framerate
            setattr(self, param_name, int(duration / duration_chunk))
        return getattr(self, param_name)

    def recognize(self, _n_audio_chunks=None, threshold=None, condition=None):
        """Function that will calculate if the VAD consider that the user is talking of a long enough duration to predict a BOT.
        Example :
        if self.silence_threshold==0.75 (percentage) and self.bot_dur==0.4 (seconds),
        It returns True if, across the frames corresponding to the last 400ms second of audio, more than 75% are containing speech.

        Returns:
            boolean : the user BOT prediction
        """
        if not _n_audio_chunks or len(self.current_input) < _n_audio_chunks:
            return False
        _n_audio_chunks = int(_n_audio_chunks)
        speech_counter = sum(
            1 for iu in self.current_input[-_n_audio_chunks:] if condition(iu)
        )
        if speech_counter >= int(threshold * _n_audio_chunks):
            return True
        return False

    def recognize_user_BOT(self):
        return self.recognize(
            _n_audio_chunks=self.get_n_audio_chunks(
                param_name="_n_bot_audio_chunks", duration=self.bot_dur
            ),
            threshold=self.silence_threshold,
            condition=lambda iu: iu.va_user,
        )

    def recognize_user_EOT(self):
        return self.recognize(
            _n_audio_chunks=self.get_n_audio_chunks(
                param_name="_n_sil_audio_chunks", duration=self.silence_dur
            ),
            threshold=self.silence_threshold,
            condition=lambda iu: not iu.va_user,
        )

    def recognize_agent_BOT(self):
        return self.current_input[-1].va_agent

    def recognize_agent_EOT(self):
        return not self.current_input[-1].va_agent

    def send_event(self, event):
        """Send message that describes the event that triggered the transition

        Args:
            event (str): event description
        """
        self.terminal_logger.info(f"event = {event}", debug=True, turn_id=self.turn_id)
        # output_iu = self.create_iu(event=event, turn_id=self.turn_id)
        output_iu = DMIU(
            creator=self,
            iuid=f"{hash(self)}:{self.iu_counter}",
            previous_iu=self._previous_iu,
            event=event,
            turn_id=self.turn_id,
        )
        um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
        self.append(um)

    def send_action(self, action):
        """Send message that describes the actions the event implies to perform

        Args:
            action (str): action description
        """
        self.terminal_logger.info(
            f"action = {action}", debug=True, turn_id=self.turn_id
        )
        # output_iu = self.create_iu(action=action, turn_id=self.turn_id)
        output_iu = DMIU(
            creator=self,
            iuid=f"{hash(self)}:{self.iu_counter}",
            previous_iu=self._previous_iu,
            action=action,
            turn_id=self.turn_id,
        )
        um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
        self.append(um)

    def send_audio_ius_last_turn(self):
        um = retico_core.UpdateMessage()
        for iu in self.audio_ius_agent_last_turn:
            # output_iu = self.create_iu(
            #     grounded_in=iu.grounded_in,
            #     raw_audio=iu.payload,
            #     nframes=self.nframes,
            #     rate=self.input_framerate,
            #     sample_width=self.sample_width,
            #     char_id=iu.char_id,
            #     clause_id=iu.clause_id,
            #     grounded_word=iu.grounded_word,
            #     word_id=iu.word_id,
            #     final=iu.final,
            #     turn_id=self.turn_id,
            #     action="repeat_last_turn",
            # )
            output_iu = DMIU(
                creator=self,
                iuid=f"{hash(self)}:{self.iu_counter}",
                grounded_in=iu.grounded_in,
                previous_iu=self._previous_iu,
                raw_audio=iu.payload,
                nframes=self.nframes,
                rate=self.input_framerate,
                sample_width=self.sample_width,
                char_id=iu.char_id,
                clause_id=iu.clause_id,
                grounded_word=iu.grounded_word,
                word_id=iu.word_id,
                final=iu.final,
                turn_id=self.turn_id,
                action="repeat_last_turn",
            )
            um.add_iu(output_iu, retico_core.UpdateType.ADD)
        self.append(um)

    def send_audio_ius(self, final=False):
        # self.terminal_logger.info(
        #     "action = process_audio", debug=True, turn_id=self.turn_id, final=final
        # )
        new_ius = self.current_input[self.buffer_pointer :]
        self.buffer_pointer = len(self.current_input)
        um = retico_core.UpdateMessage()
        ius = []
        for iu in new_ius:
            # output_iu = self.create_iu(
            #     grounded_in=self.current_input[-1],
            #     raw_audio=iu.payload,
            #     nframes=self.nframes,
            #     rate=self.input_framerate,
            #     sample_width=self.sample_width,
            #     turn_id=self.turn_id,
            #     action="process_audio",
            # )
            output_iu = DMIU(
                creator=self,
                iuid=f"{hash(self)}:{self.iu_counter}",
                previous_iu=self._previous_iu,
                grounded_in=self.current_input[-1],
                raw_audio=iu.payload,
                nframes=self.nframes,
                rate=self.input_framerate,
                sample_width=self.sample_width,
                turn_id=self.turn_id,
                action="process_audio",
            )
            ius.append((retico_core.UpdateType.ADD, output_iu))

        if final:
            for iu in self.current_input:
                # output_iu = self.create_iu(
                #     grounded_in=self.current_input[-1],
                #     raw_audio=iu.payload,
                #     nframes=self.nframes,
                #     rate=self.input_framerate,
                #     sample_width=self.sample_width,
                #     turn_id=self.turn_id,
                #     action="process_audio",
                # )
                output_iu = DMIU(
                    creator=self,
                    iuid=f"{hash(self)}:{self.iu_counter}",
                    previous_iu=self._previous_iu,
                    grounded_in=self.current_input[-1],
                    raw_audio=iu.payload,
                    nframes=self.nframes,
                    rate=self.input_framerate,
                    sample_width=self.sample_width,
                    turn_id=self.turn_id,
                    action="process_audio",
                )
                ius.append((retico_core.UpdateType.COMMIT, output_iu))
            self.current_input = []
            self.buffer_pointer = 0

        um.add_ius(ius)
        self.append(um)

    def update_current_input(self):
        self.current_input = self.current_input[
            -int(
                self.get_n_audio_chunks(
                    param_name="_n_bot_audio_chunks", duration=self.bot_dur
                )
            ) :
        ]

    def state_transition(self, source_state, destination_state):
        if source_state != destination_state:
            self.terminal_logger.info(
                f"switch state {source_state} -> {destination_state}",
                debug=True,
                turn_id=self.turn_id,
            )
        if source_state in self.policies:
            if destination_state in self.policies[source_state]:
                for action in self.policies[source_state][destination_state]:
                    # self.terminal_logger.info(
                    #     f"action : {action}", debug=True
                    # )
                    action()
        self.dialogue_state = destination_state

    def increment_turn_id(self):
        self.turn_id += 1

    def set_turn_beginning_timer(self):
        self.turn_beginning_timer = time.time()

    def check_turn_beginning_timer(self, duration_threshold=0.5):
        # if it is the beginning of the turn, set_overlap_timer
        if self.turn_beginning_timer + duration_threshold >= time.time():
            self.terminal_logger.info(
                "it is the beginning of the turn, set_overlap_timer",
                debug=True,
            )
            self.set_overlap_timer()
            self.turn_beginning_timer = -float("inf")

    def set_overlap_timer(self):
        self.overlap_timer = time.time()

    def check_overlap_timer(self, duration_threshold=1):
        if self.overlap_timer + duration_threshold >= time.time():
            self.terminal_logger.info(
                "overlap failed because both user and agent stopped talking, send repeat action to speaker module:",
                debug=True,
            )
            self.send_audio_ius_last_turn()
            self.overlap_timer = -float("inf")

    def set_repeat_timer(self, offset=3):
        self.repeat_timer = time.time() + offset

    def reset_repeat_timer(self):
        self.repeat_timer = time.time()

    def check_repeat_timer(self):
        if self.repeat_timer < time.time():
            self.increment_turn_id()
            self.terminal_logger.info(
                "repeat timer exceeded, send repeat action :",
                debug=True,
                turn_id=self.turn_id,
            )
            self.send_audio_ius_last_turn()
            self.repeat_timer = float("inf")

    def check_repeat_timer_2(self):
        if self.repeat_timer < time.time():
            self.increment_turn_id()
            self.terminal_logger.info(
                "repeat timer exceeded, send repeat action :",
                debug=True,
                turn_id=self.turn_id,
            )

            dh = self.dialogue_history.get_dialogue_history()
            last_sentence = dh[-1]["text"]
            repeat_system_prompt = (
                "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
            The teacher is teaching mathematics to the child student. \
            As the student is a child, the teacher needs to stay gentle all the time. \
            You play the role of a teacher, and your last sentence '"
                + last_sentence
                + "' had no answer from the child. Please provide a next teacher sentence that would re-engage the child in the conversation. \
            Here is the beginning of the conversation :"
            )
            previous_system_prompt = self.dialogue_history.change_system_prompt(
                repeat_system_prompt
            )
            um = retico_core.UpdateMessage()
            iu = SpeechRecognitionTurnIU(
                creator=self,
                iuid=f"{hash(self)}:{self.iu_counter}",
                previous_iu=None,
                grounded_in=None,
                predictions=["..."],
                text="...",
                stability=0.0,
                confidence=0.99,
                final=True,
                turn_id=self.turn_id,
            )
            ius = [
                (retico_core.UpdateType.ADD, iu),
                (retico_core.UpdateType.COMMIT, iu),
            ]
            um.add_ius(ius)
            self.append(um)
            self.repeat_timer = float("inf")

    def process_update(self, update_message):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L402

        Args:
            update_message (UpdateType): UpdateMessage that contains new IUs, if the IUs are ADD,
            they are added to the audio_buffer.
        """

        for iu, ut in update_message:
            if isinstance(iu, VADIU):
                if ut == retico_core.UpdateType.ADD:
                    if self.input_framerate != iu.rate:
                        raise ValueError(
                            f"input framerate differs from iu framerate : {self.input_framerate} vs {iu.rate}"
                        )
                    self.current_input.append(iu)
            if isinstance(iu, SpeakerAlignementIU):
                if iu.event == "ius_from_last_turn":
                    if len(self.audio_ius_agent_last_turn) != 0:
                        if self.audio_ius_agent_last_turn[0].turn_id != iu.turn_id:
                            self.audio_ius_agent_last_turn = [iu]
                        else:
                            self.audio_ius_agent_last_turn.append(iu)
                    else:
                        self.audio_ius_agent_last_turn = [iu]

        if self.dialogue_state == "opening":
            # 2 choices : let user engage conversation, or generate an introduction, like "hello, my name is Sara !"
            # choice 1 : waiting for the user to talk
            self.state_transition("opening", "silence_after_agent")

        elif self.dialogue_state == "user_speaking":
            user_EOT = self.recognize_user_EOT()
            agent_BOT = self.recognize_agent_BOT()
            if user_EOT:
                self.state_transition("user_speaking", "silence_after_user")
                self.send_event("user_EOT")
                self.send_audio_ius(final=True)
            else:
                self.send_audio_ius()
                if agent_BOT:
                    self.state_transition("user_speaking", "agent_overlaps_user")
                    self.send_event("agent_starts_overlaping_user")
                else:  # stay on state "user_speaking"
                    self.state_transition("user_speaking", "user_speaking")

        elif self.dialogue_state == "agent_speaking":
            agent_EOT = self.recognize_agent_EOT()
            user_BOT = self.recognize_user_BOT()
            self.update_current_input()
            if agent_EOT:
                self.state_transition("agent_speaking", "silence_after_agent")
                self.send_event("agent_EOT")
            else:
                if user_BOT:
                    self.state_transition("agent_speaking", "user_overlaps_agent")
                    self.send_event("user_barge_in")
                    # choice 1 : trigger "interruption" event
                    # choice 2 : let Turn taking handle user barge-in
                else:  # stay on state "agent_speaking"
                    self.state_transition("agent_speaking", "agent_speaking")

        elif self.dialogue_state == "silence_after_user":
            user_BOT = self.recognize_user_BOT()
            agent_BOT = self.recognize_agent_BOT()
            if user_BOT:
                if agent_BOT:
                    self.state_transition("silence_after_user", "mutual_overlap")
                    self.send_event("mutual_overlap")
                else:
                    self.state_transition("silence_after_user", "user_speaking")
                    self.send_event("user_BOT_same_turn")
            else:
                self.update_current_input()
                if agent_BOT:
                    self.state_transition("silence_after_user", "agent_speaking")
                    self.send_event("agent_BOT_new_turn")
                else:  # stay on state "silence_after_user"
                    self.state_transition("silence_after_user", "silence_after_user")

        elif self.dialogue_state == "silence_after_agent":
            user_BOT = self.recognize_user_BOT()
            agent_BOT = self.recognize_agent_BOT()
            if user_BOT:
                if agent_BOT:
                    self.state_transition("silence_after_agent", "mutual_overlap")
                    self.send_event("mutual_overlap")
                else:
                    self.state_transition("silence_after_agent", "user_speaking")
                    self.send_event("user_BOT_new_turn")
            else:
                self.update_current_input()
                if agent_BOT:
                    self.state_transition("silence_after_agent", "agent_speaking")
                    self.send_event("agent_BOT_same_turn")
                else:  # stay on state "silence_after_user"
                    self.state_transition("silence_after_agent", "silence_after_agent")

        elif self.dialogue_state == "user_overlaps_agent":
            user_EOT = self.recognize_user_EOT()
            agent_EOT = self.recognize_agent_EOT()
            if user_EOT:
                self.send_audio_ius(final=True)
                if agent_EOT:
                    self.state_transition("user_overlaps_agent", "silence_after_user")
                else:
                    self.state_transition("user_overlaps_agent", "agent_speaking")
            else:
                self.send_audio_ius()
                if agent_EOT:
                    self.state_transition("user_overlaps_agent", "user_speaking")
                else:  # stay on state "user_overlaps_agent"
                    self.state_transition("user_overlaps_agent", "user_overlaps_agent")

        elif self.dialogue_state == "agent_overlaps_user":
            user_EOT = self.recognize_user_EOT()
            agent_EOT = self.recognize_agent_EOT()
            if user_EOT:
                self.send_audio_ius(final=True)
                if agent_EOT:
                    self.state_transition("agent_overlaps_user", "silence_after_agent")
                else:
                    self.state_transition("agent_overlaps_user", "agent_speaking")
            else:
                self.send_audio_ius()
                if agent_EOT:
                    self.state_transition("agent_overlaps_user", "user_speaking")
                else:  # stay on state "agent_overlaps_user"
                    self.state_transition("agent_overlaps_user", "agent_overlaps_user")

        elif self.dialogue_state == "mutual_overlap":
            user_EOT = self.recognize_user_EOT()
            agent_EOT = self.recognize_agent_EOT()
            if user_EOT:
                self.send_audio_ius(final=True)
                if agent_EOT:
                    self.state_transition("mutual_overlap", "silence_after_agent")
                else:
                    self.state_transition("mutual_overlap", "agent_speaking")
            else:
                self.send_audio_ius()
                if agent_EOT:
                    self.state_transition("mutual_overlap", "user_speaking")
                else:  # stay on state "mutual_overlap"
                    self.state_transition("mutual_overlap", "mutual_overlap")

    ########## PIPELINE TO ALWAYS SEND VOICE TO ASR, NOT DEPENDING ON VAD_STATE
    # but to be as simple as that, it needs the BOT and EOT detection to be the same...
    # for iu, ut in update_message:
    #     if isinstance(iu, VADIU):
    #         if ut == retico_core.UpdateType.ADD:
    #             if self.input_framerate != iu.rate:
    #                 raise ValueError("input framerate differs from iu framerate")
    #             self.current_input.append(iu)
    # if self.recognize_silence():
    #     n_iu_kept = int(
    #         self.get_n_audio_chunks(
    #             param_name="_n_sil_audio_chunks", duration=self.silence_dur
    #         )
    #     )
    #     self.buffer_pointer -= len(self.current_input) - n_iu_kept
    #     self.current_input = self.current_input[-n_iu_kept:]
    # else:
    #     new_ius = self.current_input[self.buffer_pointer :]
    #     self.buffer_pointer = len(self.current_input)
    #     ius = []
    #     for iu in new_ius:
    #         output_iu = self.create_iu(
    #             grounded_in=self.current_input[-1],
    #             audio=iu.payload,
    #             nframes=self.nframes,
    #             rate=self.input_framerate,
    #             sample_width=self.sample_width,
    #             vad_state="user_turn",
    #         )
    #         ius.append((retico_core.UpdateType.ADD, output_iu))
    #     um = retico_core.UpdateMessage()
    #     um.add_ius(ius)
    #     return um
    ########################################


class VADTurnModule2(retico_core.AbstractModule):
    """A retico module using webrtcvad's Voice Activity Detection (VAD) to enhance AudioIUs with
    turn-taking informations (like user turn, silence or interruption).
    It takes AudioIUs as input and transform them into VADTurnAudioIUs by adding to it turn-taking
    informations through the IU parameter vad_state.
    It also takes TextAlignedAudioIUs as input (from the SpeakerModule), which provides information
    on when the speakers are outputting audio (when the agent is talking).

    The module considers that the current dialogue state (self.user_turn_text) can either be :
    - the user turn
    - the agent turn
    - a silence between two turns

    The transitions between the 3 dialogue states are defined as following :
    - If, while the dialogue state is a silence and the received AudioIUS are recognized as
    containing speech (VA = True), it considers that dialogue state switches to user turn, and sends
    (ADD) these IUs with vad_state = "user_turn".
    - If, while the dialogue state is user turn and a long silence is recognized (with a defined
    threshold), it considers that it is a user end of turn (EOT). It then COMMITS all IUs
    corresponding to current user turn (with vad_state = "user_turn") and dialogue state switches to
    agent turn.
    - If, while the dialogue state is agent turn, it receives the information that the SpeakerModule
    has outputted the whole agent turn (a TextAlignedAudioIU with final=True), it considers that it
    is an agent end of turn, and dialogue state switches to silence.
    - If, while the dialogue state is agent turn and before receiving an agent EOT from
    SpeakerModule, it recognize audio containing speech, it considers the current agent turn is
    interrupted by the user (user barge-in), and sends this information to the other modules to make
    the agent stop talking (by sending an empty IU with vad_state = "interruption"). Dialogue state
    then switches to user turn.

    Inputs : AudioIU, TextAlignedAudioIU

    Outputs : VADTurnAudioIU
    """

    @staticmethod
    def name():
        return "VADTurn Module"

    @staticmethod
    def description():
        return (
            "a module enhancing AudioIUs with turn-taking states using webrtcvad's VAD"
        )

    @staticmethod
    def input_ius():
        return [VADIU]

    @staticmethod
    def output_iu():
        return VADTurnAudioIU

    def __init__(
        self,
        printing=False,
        silence_dur=1,
        bot_dur=0.4,
        silence_threshold=0.75,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.printing = printing
        self.input_framerate = None
        self.channels = None
        self.sample_width = None
        self.nframes = None
        self.silence_dur = silence_dur
        self.bot_dur = bot_dur
        self.silence_threshold = silence_threshold
        self._n_sil_audio_chunks = None
        self._n_bot_audio_chunks = None
        self.vad_state = False
        self.user_turn = False
        self.user_turn_text = "no speaker"
        self.buffer_pointer = 0

    def get_n_audio_chunks(self, param_name, duration):
        """Returns the number of audio chunks containing speech needed in the audio buffer to have a BOT (beginning of turn)
        (ie. to how many audio_chunk correspond self.bot_dur)

        Returns:
            int: the number of audio chunks corresponding to the duration of self.bot_dur.
        """
        if not getattr(self, param_name):
            if len(self.current_input) == 0:
                return None
            first_iu = self.current_input[0]
            self.input_framerate = first_iu.rate
            self.nframes = first_iu.nframes
            self.sample_width = first_iu.sample_width
            # nb frames in each audio chunk
            nb_frames_chunk = len(first_iu.payload) / 2
            # duration of 1 audio chunk
            duration_chunk = nb_frames_chunk / self.input_framerate
            setattr(self, param_name, int(duration / duration_chunk))
        return getattr(self, param_name)

    def recognize(self, _n_audio_chunks=None, threshold=None, condition=None):
        """Function that will calculate if the VAD consider that the user is talking of a long enough duration to predict a BOT.
        Example :
        if self.silence_threshold==0.75 (percentage) and self.bot_dur==0.4 (seconds),
        It returns True if, across the frames corresponding to the last 400ms second of audio, more than 75% are containing speech.

        Returns:
            boolean : the user BOT prediction
        """
        if not _n_audio_chunks or len(self.current_input) < _n_audio_chunks:
            return False
        _n_audio_chunks = int(_n_audio_chunks)
        speech_counter = sum(
            1 for iu in self.current_input[-_n_audio_chunks:] if condition
        )
        if speech_counter >= int(threshold * _n_audio_chunks):
            return True
        return False

    def recognize_bot(self):
        return self.recognize(
            _n_audio_chunks=self.get_n_audio_chunks(
                param_name="_n_bot_audio_chunks", duration=self.bot_dur
            ),
            threshold=self.silence_threshold,
            condition=lambda iu: iu.va_user,
        )

    def recognize_silence(self):
        return self.recognize(
            _n_audio_chunks=self.get_n_audio_chunks(
                param_name="_n_sil_audio_chunks", duration=self.silence_dur
            ),
            threshold=self.silence_threshold,
            condition=lambda iu: not iu.va_user,
        )

    def process_update(self, update_message):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L402

        Args:
            update_message (UpdateType): UpdateMessage that contains new IUs, if the IUs are ADD,
            they are added to the audio_buffer.
        """
        for iu, ut in update_message:
            if isinstance(iu, VADIU):
                if ut == retico_core.UpdateType.ADD:
                    if self.input_framerate != iu.rate:
                        raise ValueError(
                            f"input framerate differs from iu framerate : {self.input_framerate} vs {iu.rate}"
                        )
                    self.current_input.append(iu)

        if self.user_turn_text == "agent":
            # It is not a user turn, The agent could be speaking, or it could have finished speaking.
            # We are listenning for potential user beginning of turn (bot).
            bot = self.recognize_bot()
            if bot:
                # user wasn't talking, but he starts talking
                # A bot has been detected, we'll :
                # - set the user_turn parameter as True
                # - Take only the end of the audio_buffer, to remove the useless audio
                # - Send a INTERRUPTION IU to all modules to make them stop generating new data (if the agent is talking, he gets interrupted by the user)
                # self.user_turn = True
                self.user_turn_text = "user_turn"
                self.buffer_pointer = 0

                output_iu = self.create_iu(
                    grounded_in=self.current_input[-1],
                    vad_state="interruption",
                )
                self.terminal_logger.info("interruption")
                self.file_logger.info("interruption")

                return retico_core.UpdateMessage.from_iu(
                    output_iu, retico_core.UpdateType.ADD
                )

            else:
                # print("SILENCE")
                # user wasn't talkin, and stays quiet
                # No bot has been detected, we'll
                # - empty the audio buffer to remove useless audio
                self.current_input = self.current_input[
                    -int(
                        self.get_n_audio_chunks(
                            param_name="_n_bot_audio_chunks", duration=self.bot_dur
                        )
                    ) :
                ]
                # print("remove from audio buffer")

        # else:
        elif self.user_turn_text == "user_turn":
            # It is user turn, we are listenning for a long enough silence, which would be analyzed as a user EOT.
            silence = self.recognize_silence()
            if not silence:
                # print("TALKING")
                # User was talking, and is still talking
                # no user EOT has been predicted, we'll :
                # - Send all new IUs containing audio corresponding to parts of user sentence to the whisper module to generate a new transcription hypothesis.
                # print("len(self.audio_buffer) = ", len(self.audio_buffer))
                # print("self.buffer_pointer = ", self.buffer_pointer)
                new_ius = self.current_input[self.buffer_pointer :]
                self.buffer_pointer = len(self.current_input)
                ius = []
                for iu in new_ius:
                    output_iu = self.create_iu(
                        grounded_in=self.current_input[-1],
                        audio=iu.payload,
                        nframes=self.nframes,
                        rate=self.input_framerate,
                        sample_width=self.sample_width,
                        vad_state="user_turn",
                    )
                    ius.append((retico_core.UpdateType.ADD, output_iu))
                um = retico_core.UpdateMessage()
                um.add_ius(ius)
                return um

            else:
                self.terminal_logger.info("user_EOT")
                self.file_logger.info("user_EOT")
                # User was talking, but is not talking anymore (a >700ms silence has been observed)
                # a user EOT has been predicted, we'll :
                # - ADD additional IUs if there is some (sould not happen)
                # - COMMIT all audio in audio_buffer to generate the transcription from the full user sentence using ASR.
                # - set the user_turn as False
                # - empty the audio buffer
                ius = []

                # Add the last AudioIU if there is additional audio since last update_message (should not happen)
                if self.buffer_pointer != len(self.current_input) - 1:
                    for iu in self.current_input[-self.buffer_pointer :]:
                        output_iu = self.create_iu(
                            grounded_in=self.current_input[-1],
                            audio=iu.payload,
                            nframes=self.nframes,
                            rate=self.input_framerate,
                            sample_width=self.sample_width,
                            vad_state="user_turn",
                        )
                        ius.append((retico_core.UpdateType.ADD, output_iu))

                for iu in self.current_input:
                    output_iu = self.create_iu(
                        grounded_in=self.current_input[-1],
                        audio=iu.payload,
                        nframes=self.nframes,
                        rate=self.input_framerate,
                        sample_width=self.sample_width,
                        vad_state="user_turn",
                    )
                    ius.append((retico_core.UpdateType.COMMIT, output_iu))

                um = retico_core.UpdateMessage()
                um.add_ius(ius)
                self.user_turn_text = "agent"
                self.current_input = []
                self.buffer_pointer = 0
                return um

        elif self.user_turn_text == "no speaker":
            # nobody is speaking, we are waiting for user to speak.
            # We are listenning for potential user beginning of turn (bot).
            bot = self.recognize_bot()
            if bot:
                self.terminal_logger.info("user_BOT")
                self.file_logger.info("user_BOT")
                # user wasn't talking, but he starts talking
                # A bot has been detected, we'll :
                # - set the user_turn parameter as True
                # - Take only the end of the audio_buffer, to remove the useless audio
                # - Send a INTERRUPTION IU to all modules to make them stop generating new data (if the agent is talking, he gets interrupted by the user)
                # self.user_turn = True
                self.user_turn_text = "user_turn"
            else:
                # user wasn't talkin, and stays quiet
                # No bot has been detected, we'll
                # - empty the audio buffer to remove useless audio
                self.current_input = self.current_input[
                    -int(
                        self.get_n_audio_chunks(
                            param_name="_n_bot_audio_chunks", duration=self.bot_dur
                        )
                    ) :
                ]
