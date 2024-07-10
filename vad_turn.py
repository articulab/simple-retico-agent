"""
VAD Module
==================

This module uses webrtcvad's Voice Activity Detection (VAD) to enhance AudioIUs with turn-taking
informations (like user_turn, silence or interruption).
It takes AudioIUs as input and transform them into AudioVADIUs by adding to it turn-taking
informations through the IU parameter vad_state.
It also takes TurnAudioIUs as input (from the SpeakerModule), which provides information on when the
speakers are outputting audio (when the agent is talking).

The module considers that the current dialogue state (self.user_turn_text) can either be : 
- the user turn
- the agent turn
- a silence between two turns

The transitions between the 3 dialogue states are defined as following :
- If, while the dialogue state is a silence and the received AudioIUS are recognized as containing 
speech (VA = True), it considers that dialogue state switches to user turn, and sends (ADD) these
IUs with vad_state = "user_turn".
- If, while the dialogue state is user turn and a long silence is recognized (with a defined
threshold), it considers that it is a user end of turn (EOT). It then COMMITS all IUs corresponding
to current user turn (with vad_state = "user_turn") and dialogue state switches to agent turn.
- If, while the dialogue state is agent turn, it receives the information that the SpeakerModule has
outputted the whole agent turn (a TurnAudioIU with final=True), it considers that it is an agent end
of turn, and dialogue state switches to silence.
- If, while the dialogue state is agent turn and before receiving an agent EOT from SpeakerModule,
it recognize audio containing speech, it considers the current agent turn is interrupted by the user
(user barge-in), and sends this information to the other modules to make the agent stop talking (by 
sending an empty IU with vad_state = "interruption"). Dialogue state then switches to user turn.

Inputs : AudioIU, TurnAudioIU

Outputs : AudioVADIU
"""

import retico_core
from retico_core.audio import AudioIU
import pydub
import webrtcvad

from utils import *


class VADTurnModule(retico_core.AbstractModule):
    """a retico module using webrtcvad's Voice Activity Detection (VAD) to enhance AudioIUs with
    turn-taking informations (like user turn, silence or interruption).
    It takes AudioIUs as input and transform them into AudioVADIUs by adding to it turn-taking
    informations through the IU parameter vad_state.
    It also takes TurnAudioIUs as input (from the SpeakerModule), which provides information on when
    the speakers are outputting audio (when the agent is talking).

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
    has outputted the whole agent turn (a TurnAudioIU with final=True), it considers that it is an
    agent end of turn, and dialogue state switches to silence.
    - If, while the dialogue state is agent turn and before receiving an agent EOT from
    SpeakerModule, it recognize audio containing speech, it considers the current agent turn is
    interrupted by the user (user barge-in), and sends this information to the other modules to make
    the agent stop talking (by sending an empty IU with vad_state = "interruption"). Dialogue state
    then switches to user turn.

    Inputs : AudioIU, TurnAudioIU

    Outputs : AudioVADIU
    """

    @staticmethod
    def name():
        return "VAD Turn Module"

    @staticmethod
    def description():
        return (
            "a module enhancing AudioIUs with turn-taking states using webrtcvad's VAD"
        )

    @staticmethod
    def input_ius():
        return [AudioIU, TurnAudioIU]

    @staticmethod
    def output_iu():
        return AudioVADIU

    def __init__(
        self,
        printing=False,
        log_file="vad.csv",
        log_folder="logs/test/16k/Recording (1)/demo",
        target_framerate=16000,
        input_framerate=44100,
        channels=1,
        sample_width=2,
        silence_dur=1,
        bot_dur=0.4,
        silence_threshold=0.75,
        vad_aggressiveness=3,
        frame_length=0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.printing = printing
        self.log_file = manage_log_folder(log_folder, log_file)
        self.time_logs_buffer = []

        self.target_framerate = target_framerate
        self.input_framerate = input_framerate
        self.channels = channels
        self.sample_width = sample_width
        self.frame_length = frame_length
        self.chunk_size = round(self.target_framerate * self.frame_length)
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.silence_dur = silence_dur
        self._n_sil_audio_chunks = None
        self.silence_threshold = silence_threshold
        self.bot_dur = bot_dur
        self._n_bot_audio_chunks = None
        self.vad_state = False
        self.user_turn = False
        self.user_turn_text = "no speaker"
        self.audio_buffer = []
        self.buffer_pointer = 0

        # latency logs params
        self.first_time = True
        self.first_time_stop = False

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

    def add_audio(self, audio):
        """Resamples and adds the audio chunk received from the microphone to the audio buffer.

        Args:
            audio (bytes): the audio chunk received from the microphone.
        """
        audio = self.resample_audio(audio)
        self.audio_buffer.append(audio)

    def get_n_sil_audio_chunks(self):
        """Returns the number of silent audio chunks needed in the audio buffer to have an EOS
        (ie. to how many audio_chunk correspond self.silence_dur)

        Returns:
            int: the number of audio chunks corresponding to the duration of self.silence_dur.
        """
        if not self._n_sil_audio_chunks:
            if len(self.audio_buffer) == 0:
                return None
            # nb frames in each audio chunk
            nb_frames_chunk = len(self.audio_buffer[0]) / 2
            # duration of 1 audio chunk
            duration_chunk = nb_frames_chunk / self.target_framerate
            self._n_sil_audio_chunks = int(self.silence_dur / duration_chunk)
        return self._n_sil_audio_chunks

    def get_n_bot_audio_chunks(self):
        """Returns the number of audio chunks containing speech needed in the audio buffer to have a BOT (beginning of turn)
        (ie. to how many audio_chunk correspond self.bot_dur)

        Returns:
            int: the number of audio chunks corresponding to the duration of self.bot_dur.
        """
        if not self._n_bot_audio_chunks:
            if len(self.audio_buffer) == 0:
                return None
            # nb frames in each audio chunk
            nb_frames_chunk = len(self.audio_buffer[0]) / 2
            # duration of 1 audio chunk
            duration_chunk = nb_frames_chunk / self.target_framerate
            self._n_bot_audio_chunks = int(self.bot_dur / duration_chunk)
        return self._n_bot_audio_chunks

    def recognize_silence(self):
        """Function that will calculate if the ASR consider that there is a silence long enough to be a user EOS.
        Example :
        if self.silence_threshold==0.75 (percentage) and self.silence_dur==1 (seconds),
        It returns True if, across the frames corresponding to the last 1 second of audio, more than 75% are considered as silence by the vad.

        Returns:
            boolean : the user EOS prediction
        """

        _n_sil_audio_chunks = self.get_n_sil_audio_chunks()
        if not _n_sil_audio_chunks or len(self.audio_buffer) < _n_sil_audio_chunks:
            # print("silence 0")
            return True
        _n_sil_audio_chunks = int(_n_sil_audio_chunks)
        silence_counter = sum(
            1
            for a in self.audio_buffer[-_n_sil_audio_chunks:]
            if not self.vad.is_speech(a, self.target_framerate)
        )
        if silence_counter >= int(self.silence_threshold * _n_sil_audio_chunks):
            # print("silence 1")
            return True
        # print("silence 2")
        return False

    def recognize_silence_2(self):
        """Function that will calculate if the ASR consider that there is a silence long enough to be a user EOS.
        Example :
        if self.silence_threshold==0.75 (percentage) and self.silence_dur==1 (seconds),
        It returns True if, across the frames corresponding to the last 1 second of audio, more than 75% are considered as silence by the vad.

        Returns:
            boolean : the user EOS prediction
        """

        _n_sil_audio_chunks = self.get_n_sil_audio_chunks()
        if not _n_sil_audio_chunks or len(self.audio_buffer) < _n_sil_audio_chunks:
            # print("silence 0")
            return False
        _n_sil_audio_chunks = int(_n_sil_audio_chunks)
        silence_counter = sum(
            1
            for a in self.audio_buffer[-_n_sil_audio_chunks:]
            if not self.vad.is_speech(a, self.target_framerate)
        )
        if silence_counter >= int(self.silence_threshold * _n_sil_audio_chunks):
            # print("silence 1")
            return True
        # print("silence 2")
        return False

    def recognize_bot(self):
        """Function that will calculate if the VAD consider that the user is talking of a long enough duration to predict a BOT.
        Example :
        if self.silence_threshold==0.75 (percentage) and self.bot_dur==0.4 (seconds),
        It returns True if, across the frames corresponding to the last 400ms second of audio, more than 75% are containing speech.

        Returns:
            boolean : the user BOT prediction
        """
        _n_bot_audio_chunks = self.get_n_bot_audio_chunks()
        if not _n_bot_audio_chunks or len(self.audio_buffer) < _n_bot_audio_chunks:
            return False
        _n_bot_audio_chunks = int(_n_bot_audio_chunks)
        speech_counter = sum(
            1
            for a in self.audio_buffer[-_n_bot_audio_chunks:]
            if self.vad.is_speech(a, self.target_framerate)
        )
        if speech_counter >= int(self.silence_threshold * _n_bot_audio_chunks):
            return True
        return False

    def process_update(self, update_message):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L402

        Args:
            update_message (UpdateType): UpdateMessage that contains new IUs, if the IUs are ADD,
            they are added to the audio_buffer.
        """
        lastest_iu = None
        for iu, ut in update_message:
            if isinstance(iu, TurnAudioIU):
                if ut == retico_core.UpdateType.ADD:
                    if iu.final:
                        # print("VADTURN : agent stopped talking")
                        # self.user_turn = True
                        self.user_turn_text = "no speaker"
            elif isinstance(iu, AudioIU):
                if ut == retico_core.UpdateType.ADD:
                    if self.input_framerate != iu.rate:
                        raise Exception("input framerate differs from iu framerate")
                    self.add_audio(iu.raw_audio)
                    lastest_iu = iu

        # if not self.user_turn:
        if self.user_turn_text == "agent":
            # It is not a user turn, The agent could be speaking, or it could have finished speaking.
            # We are listenning for potential user beginning of turn (bot).
            bot = self.recognize_bot()
            if bot:
                if self.printing:
                    print("VAD INTERRUPTION")
                # user wasn't talking, but he starts talking
                # A bot has been detected, we'll :
                # - set the user_turn parameter as True
                # - Take only the end of the audio_buffer, to remove the useless audio
                # - Send a INTERRUPTION IU to all modules to make them stop generating new data (if the agent is talking, he gets interrupted by the user)
                # self.user_turn = True
                self.user_turn_text = "user"

                # self.audio_buffer = self.audio_buffer[
                #     -int(self.get_n_bot_audio_chunks()) :
                # ]
                # print("BOT remove from audio buffer")

                output_iu = self.create_iu(lastest_iu)
                output_iu.set_data(vad_state="interruption")
                # output_iu.set_data(
                #     audio=audio,
                #     chunk_size=self.chunk_size,
                #     rate=self.rate,
                #     sample_width=self.sample_width,
                #     vad_state="interruption",
                # )

                return retico_core.UpdateMessage.from_iu(
                    output_iu, retico_core.UpdateType.ADD
                )
            else:
                # print("SILENCE")
                # user wasn't talkin, and stays quiet
                # No bot has been detected, we'll
                # - empty the audio buffer to remove useless audio
                self.audio_buffer = self.audio_buffer[
                    -int(self.get_n_bot_audio_chunks()) :
                ]
                # print("remove from audio buffer")

        # else:
        elif self.user_turn_text == "user":
            # It is user turn, we are listenning for a long enough silence, which would be analyzed as a user EOT.
            silence = self.recognize_silence_2()
            if not silence:
                # print("TALKING")
                # User was talking, and is still talking
                # no user EOT has been predicted, we'll :
                # - Send all new IUs containing audio corresponding to parts of user sentence to the whisper module to generate a new transcription hypothesis.
                # print("len(self.audio_buffer) = ", len(self.audio_buffer))
                # print("self.buffer_pointer = ", self.buffer_pointer)
                new_audio = self.audio_buffer[self.buffer_pointer :]
                self.buffer_pointer = len(self.audio_buffer)
                # print("new_audio = ", len(new_audio))
                # print("new self.buffer_pointer = ", self.buffer_pointer)
                ius = []
                for audio in new_audio:
                    output_iu = self.create_iu(lastest_iu)
                    output_iu.set_data(
                        audio=audio,
                        chunk_size=self.chunk_size,
                        rate=self.target_framerate,
                        sample_width=self.sample_width,
                        vad_state="user_turn",
                    )
                    ius.append((retico_core.UpdateType.ADD, output_iu))
                um = retico_core.UpdateMessage()
                um.add_ius(ius)
                return um

            else:
                if self.printing:
                    print("VAD EOT")
                # User was talking, but is not talking anymore (a >700ms silence has been observed)
                # a user EOT has been predicted, we'll :
                # - ADD additional IUs if there is some (sould not happen)
                # - COMMIT all audio in audio_buffer to generate the transcription from the full user sentence using ASR.
                # - set the user_turn as False
                # - empty the audio buffer
                ius = []

                # Add the last AudioIU if there is additional audio since last update_message (should not happen)
                if self.buffer_pointer != len(self.audio_buffer) - 1:
                    for audio in self.audio_buffer[-self.buffer_pointer :]:
                        output_iu = self.create_iu(lastest_iu)
                        # output_iu.set_data(audio=audio, vad_state="user_turn")
                        output_iu.set_data(
                            audio=audio,
                            chunk_size=self.chunk_size,
                            rate=self.target_framerate,
                            sample_width=self.sample_width,
                            vad_state="user_turn",
                        )
                        # ius.append((output_iu, retico_core.UpdateType.ADD))
                        ius.append((retico_core.UpdateType.ADD, output_iu))

                for audio in self.audio_buffer:
                    output_iu = self.create_iu(lastest_iu)
                    # output_iu.set_data(audio=audio, vad_state="user_turn")
                    output_iu.set_data(
                        audio=audio,
                        chunk_size=self.chunk_size,
                        rate=self.target_framerate,
                        sample_width=self.sample_width,
                        vad_state="user_turn",
                    )
                    ius.append((retico_core.UpdateType.COMMIT, output_iu))
                    # ius.append((output_iu, retico_core.UpdateType.COMMIT))

                um = retico_core.UpdateMessage()
                um.add_ius(ius)
                # self.user_turn = False
                self.user_turn_text = "agent"
                self.audio_buffer = []
                # print("reset audio buffer")
                return um

        elif self.user_turn_text == "no speaker":
            # nobody is speaking, we are waiting for user to speak.
            # We are listenning for potential user beginning of turn (bot).
            bot = self.recognize_bot()
            if bot:
                if self.printing:
                    print("VAD BOT")
                # user wasn't talking, but he starts talking
                # A bot has been detected, we'll :
                # - set the user_turn parameter as True
                # - Take only the end of the audio_buffer, to remove the useless audio
                # - Send a INTERRUPTION IU to all modules to make them stop generating new data (if the agent is talking, he gets interrupted by the user)
                # self.user_turn = True
                self.user_turn_text = "user"
            else:
                # print("SILENCE")
                # user wasn't talkin, and stays quiet
                # No bot has been detected, we'll
                # - empty the audio buffer to remove useless audio
                self.audio_buffer = self.audio_buffer[
                    -int(self.get_n_bot_audio_chunks()) :
                ]
