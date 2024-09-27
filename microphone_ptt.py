"""
MicrophonePTTModule
==================

This module provides push-to-talk capabilities to the classic retico MicrophoneModule
which captures audio signal from the microphone and chunks the audio signal into AudioIUs.
"""

import queue
import keyboard
import pyaudio

import retico_core
from retico_core.audio import MicrophoneModule


class MicrophonePTTModule(MicrophoneModule):
    """A modules overrides the MicrophoneModule which captures audio signal from the microphone and chunks the audio signal into AudioIUs.
    The addition of this module is the introduction of the push-to-talk capacity : the microphone's audio signal is captured only while the M key is pressed.
    """

    def callback(self, in_data, frame_count, time_info, status):
        """The callback function that gets called by pyaudio.

        Args:
            in_data (bytes[]): The raw audio that is coming in from the
                microphone
            frame_count (int): The number of frames that are stored in in_data
        """
        if keyboard.is_pressed("m"):
            self.audio_buffer.put(in_data)
        else:
            self.audio_buffer.put(b"\x00" * self.sample_width * self.chunk_size)
        return (in_data, pyaudio.paContinue)

    def process_update(self, _):
        """overrides MicrophoneModule : https://github.com/retico-team/retico-core/blob/main/retico_core/audio.py#202

        Returns:
            UpdateMessage: list of AudioIUs produced from the microphone's audio signal.
        """
        if not self.audio_buffer:
            return None
        try:
            sample = self.audio_buffer.get(timeout=1.0)
        except queue.Empty:
            return None
        # output_iu = self.create_iu()
        # output_iu.set_audio(sample, self.chunk_size, self.rate, self.sample_width)
        output_iu = self.create_iu(
            raw_audio=sample,
            nframes=self.chunk_size,
            rate=self.rate,
            sample_width=self.sample_width,
        )
        return retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
