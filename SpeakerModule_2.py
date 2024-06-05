"""
SpeakerModule_2
==================

This module outputs the audio signal contained in the AudioIUs by the computer's speakers.
"""
import datetime
import pyaudio
import retico_core
import platform
from retico_core.audio import AudioIU, SpeakerModule
from utils import *

from utils import *

class SpeakerModule_2(SpeakerModule):
    """A module that consumes AudioIUs of arbitrary size and outputs them to the
    speakers of the machine. When a new IU is incoming, the module blocks as
    long as the current IU is being played."""

    def __init__(
        self,
        rate=44100,
        sample_width=2,
        channels=1,
        use_speaker="both",
        device_index=None,
        log_file="speaker.csv",
        log_folder="logs/test/16k/Recording (1)/demo",
        **kwargs
    ):
        super().__init__(
            rate=rate,
            sample_width=sample_width,
            use_speaker=use_speaker,
            device_index=device_index,
            **kwargs
        )
        self.channels=channels
        self.log_file = manage_log_folder(log_folder, log_file)
        self.time_logs_buffer = []
        self.first_time = True

    def process_update(self, update_message):
        """overrides SpeakerModule : https://github.com/retico-team/retico-core/blob/main/retico_core/audio.py#L282

        overrides SpeakerModule's process_update to save logs.
        """
        # TODO: replace this method by an actual way of knowing what is the starting and ending time where the speaker is active (actually outputs time, and not receive messages).
        if self.first_time:
            self.time_logs_buffer.append(
                ["Start", datetime.datetime.now().strftime("%T.%f")[:-3]]
            )
            self.first_time = False
        else:
            self.time_logs_buffer.append(
                ["Stop", datetime.datetime.now().strftime("%T.%f")[:-3]]
            )
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.ADD:
                self.stream.write(bytes(iu.raw_audio))
        return None

    def setup(self):
        """overrides SpeakerModule : https://github.com/retico-team/retico-core/blob/main/retico_core/audio.py#L288

        overrides to give correct CHANNEL parameter"""
        p = self._p

        if platform.system() == "Darwin":
            if self.use_speaker == "left":
                stream_info = pyaudio.PaMacCoreStreamInfo(channel_map=(0, -1))
            elif self.use_speaker == "right":
                stream_info = pyaudio.PaMacCoreStreamInfo(channel_map=(-1, 0))
            else:
                stream_info = pyaudio.PaMacCoreStreamInfo(channel_map=(0, 0))
        else:
            stream_info = None

        self.stream = p.open(
            format=p.get_format_from_width(self.sample_width),
            channels=self.channels,
            rate=self.rate,
            input=False,
            output_host_api_specific_stream_info=stream_info,
            output=True,
            output_device_index=self.device_index,
        )
        
    def shutdown(self):
        """overrides SpeakerModule : https://github.com/retico-team/retico-core/blob/main/retico_core/audio.py#L312
        
        Write logs and close the audio stream."""
        write_logs(
            self.log_file,
            self.time_logs_buffer,
        )
        self.stream.stop_stream()
        self.stream.close()
        self.stream = None
