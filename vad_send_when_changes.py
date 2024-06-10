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
        return VADStateIU

    def __init__(
        self,
        printing=False,
        log_file="vad.csv",
        log_folder="logs/test/16k/Recording (1)/demo",
        vad_aggressiveness=3,
        target_framerate=16000,
        input_framerate=44100,
        channels=1,
        sample_width=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_framerate = target_framerate
        self.input_framerate = input_framerate
        self.channels = channels
        self.sample_width = sample_width
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.printing = printing
        self.log_file = manage_log_folder(log_folder, log_file)
        self.time_logs_buffer = []
        self.vad_state = False

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
        ius = []
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            if self.input_framerate != iu.rate:
                raise Exception("input framerate differs from iu framerate")
            is_speech = self.vad.is_speech(
                self.resample_audio(iu.raw_audio), self.target_framerate
            )
            if is_speech != self.vad_state:
                if is_speech:
                    print("Someone starts talking")
                else:
                    print("Someone stopped talking")
                self.vad_state = is_speech
                output_iu = self.create_iu(iu)
                output_iu.set_vad_state(self.vad_state)
                ius.append((output_iu, retico_core.UpdateType.ADD))
        um = retico_core.UpdateMessage()
        um.add_ius(ius)
        return um
