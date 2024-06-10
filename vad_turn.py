"""
VAD Module
==================

This module provides Voice Activity Detection (VAD) using a webrtcvad.
Only send an update message if the vad_state changes.
"""

import retico_core
from retico_core.audio import AudioIU
import pydub
import webrtcvad

from utils import *


class VADStateIU(retico_core.abstract.IncrementalUnit):
    """An Incremental IU describing the vad state (if someone is talking or not).

    Attributes:
        creator (AbstractModule): The module that created this IU
        previous_iu (IncrementalUnit): A link to the IU created before the
            current one.
        grounded_in (IncrementalUnit): A link to the IU this IU is based on.
        created_at (float): The UNIX timestamp of the moment the IU is created.
        vad_state (bool): The vad state.
    """

    @staticmethod
    def type():
        return "VAD State IU"

    def __init__(
        self,
        creator=None,
        iuid=0,
        previous_iu=None,
        grounded_in=None,
        vad_state=None,
        **kwargs,
    ):
        super().__init__(
            creator=creator,
            iuid=iuid,
            previous_iu=previous_iu,
            grounded_in=grounded_in,
            payload=vad_state,
        )
        self.vad_state = vad_state

    def set_vad_state(self, vad_state):
        """Sets the vad_state"""
        self.payload = vad_state
        self.vad_state = vad_state


class AudioVADIU(retico_core.audio.AudioIU):
    """An Incremental IU describing the vad state (if someone is talking or not).

    Attributes:
        creator (AbstractModule): The module that created this IU
        previous_iu (IncrementalUnit): A link to the IU created before the
            current one.
        grounded_in (IncrementalUnit): A link to the IU this IU is based on.
        created_at (float): The UNIX timestamp of the moment the IU is created.
        vad_state (bool): The vad state.
    """

    @staticmethod
    def type():
        return "Audio VAD IU"

    def __init__(
        self,
        creator=None,
        iuid=0,
        previous_iu=None,
        grounded_in=None,
        audio=None,
        vad_state=None,
        rate=None,
        nframes=None,
        sample_width=None,
        **kwargs,
    ):
        super().__init__(
            creator=creator,
            iuid=iuid,
            previous_iu=previous_iu,
            grounded_in=grounded_in,
            payload=audio,
            raw_audio=audio,
            rate=rate,
            nframes=nframes,
            sample_width=sample_width,
        )
        self.vad_state = vad_state

    def set_data(self, audio, vad_state):
        """Sets the vad_state"""
        self.payload = audio
        self.raw_audio = audio
        self.vad_state = vad_state


class VADSendWhenChangesModule(retico_core.AbstractModule):
    """A retico module that provides Voice Activity Detection (VAD) using a webrtcvad.

    Inputs : AudioIU

    Outputs : VADStateIU
    """

    @staticmethod
    def name():
        return "webrtcvad Module"

    @staticmethod
    def description():
        return "A module that recognizes if someone is talking using webrtcvad."

    @staticmethod
    def input_ius():
        return [AudioIU]

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
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.silence_dur = silence_dur
        self._n_sil_audio_chunks = None
        self.silence_threshold = silence_threshold
        self.bot_dur = bot_dur
        self._n_bot_audio_chunks = None
        self.vad_state = False
        self.user_turn = False
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
            return True
        _n_sil_audio_chunks = int(_n_sil_audio_chunks)
        silence_counter = sum(
            1
            for a in self.audio_buffer[-_n_sil_audio_chunks:]
            if not self.vad.is_speech(a, self.target_framerate)
        )
        if silence_counter >= int(self.silence_threshold * _n_sil_audio_chunks):
            return True
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
            return True
        _n_bot_audio_chunks = int(_n_bot_audio_chunks)
        silence_counter = sum(
            1
            for a in self.audio_buffer[-_n_bot_audio_chunks:]
            if not self.vad.is_speech(a, self.target_framerate)
        )
        if silence_counter >= int(self.silence_threshold * _n_bot_audio_chunks):
            return True
        return False

    def process_update(self, update_message):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L402

        Args:
            update_message (UpdateType): UpdateMessage that contains new IUs, if the IUs are ADD,
            they are added to the audio_buffer.
        """
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            if self.input_framerate != iu.rate:
                raise Exception("input framerate differs from iu framerate")

            self.add_audio(iu.raw_audio)

        if not self.user_turn:
            # It is not a user turn, The agent could be speaking, or it could have finished speaking.
            # We are listenning for potential user beginning of turn (bot).
            bot = self.recognize_bot()
            if bot:
                # user wasn't talking, but he starts talking
                # A bot has been detected, we'll :
                # - set the user_turn parameter as True
                # - Take only the end of the audio_buffer, to remove the useless audio
                # - Send a INTERRUPTION IU to all modules to make them stop generating new data (if the agent is talking, he gets interrupted by the user)
                self.user_turn = True
                self.audio_buffer = self.audio_buffer[
                    -int(self.get_n_bot_audio_chunks()) :
                ]

                output_iu = self.create_iu(iu)
                output_iu.set_data(audio=None, vad_state="interruption")
                return retico_core.UpdateMessage.from_iu(
                    output_iu, retico_core.UpdateType.ADD
                )
            else:
                # user wasn't talkin, and stays quiet
                # No bot has been detected, we'll
                # - empty the audio buffer to remove useless audio
                self.audio_buffer = self.audio_buffer[
                    -int(self.get_n_bot_audio_chunks()) :
                ]
        else:
            # It is user turn, we are listenning for a long enough silence, which would be analyzed as a user EOT.
            silence = self.recognize_silence()
            if not silence:
                # User was talking, and is still talking
                # no user EOT has been predicted, we'll :
                # - Send all new IUs containing audio corresponding to parts of user sentence to the whisper module to generate a new transcription hypothesis.
                new_audio = self.audio_buffer[-self.buffer_pointer :]
                self.buffer_pointer = len(self.audio_buffer) - 1
                ius = []
                for audio in new_audio:
                    output_iu = self.create_iu(iu)
                    output_iu.set_data(audio=audio, vad_state="user_turn")
                    ius.append((output_iu, retico_core.UpdateType.ADD))
                um = retico_core.UpdateMessage()
                um.add_ius(ius)
                return um

            else:
                # User was talking, but is not talking anymore (a >700ms silence has been observed)
                # a user EOT has been predicted, we'll :
                # - Send all IUs containing the audio corresponding to the full user sentence to the whisper module to generate the transcription.
                # - set the user_turn as False
                # - empty the audio buffer
                ius = []
                for audio in self.audio_buffer:
                    output_iu = self.create_iu(iu)
                    output_iu.set_data(audio=audio, vad_state="user_turn")
                    ius.append((output_iu, retico_core.UpdateType.COMMIT))
                um = retico_core.UpdateMessage()
                um.add_ius(ius)
                self.user_turn = False
                self.audio_buffer = []
                return um
