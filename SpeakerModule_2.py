# import time
# import wave
import csv
import datetime
import pyaudio
import retico_core
import platform
from retico_core.audio import AudioIU

from utils import *

CHANNELS = 1


class SpeakerModule_2(retico_core.AbstractConsumingModule):
    """A module that consumes AudioIUs of arbitrary size and outputs them to the
    speakers of the machine. When a new IU is incoming, the module blocks as
    long as the current IU is being played."""

    @staticmethod
    def name():
        return "Speaker Module"

    @staticmethod
    def description():
        return "A consuming module that plays audio from speakers."

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return None

    def __init__(
        self,
        rate=44100,
        sample_width=2,
        use_speaker="both",
        device_index=None,
        log_file="speaker.csv",
        log_folder="logs/test/16k/Recording (1)/demo",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.rate = rate
        self.sample_width = sample_width
        self.use_speaker = use_speaker

        self._p = pyaudio.PyAudio()

        if device_index is None:
            device_index = self._p.get_default_output_device_info()["index"]
        self.device_index = device_index

        self.stream = None
        self.time = None
        # logs
        self.log_file = manage_log_folder(log_folder, log_file)
        self.first_time = True

    def process_update(self, update_message):
        if self.first_time:
            write_logs(
                self.log_file,
                [["Start", datetime.datetime.now().strftime("%T.%f")[:-3]]],
            )
            self.first_time = False
        else:
            write_logs(
                self.log_file,
                [["Stop", datetime.datetime.now().strftime("%T.%f")[:-3]]],
            )
        # silence = chr(0) * self.chunk * self.channels * 2
        # cpt_total = 0
        # cpt = 0
        for iu, ut in update_message:
            # cpt_total += 1
            if ut == retico_core.UpdateType.ADD:
                # cpt += 1
                self.stream.write(bytes(iu.raw_audio))
        # print("\n\n" + str(cpt_total))
        # print(cpt)
        # free = self.stream.get_write_available()
        # tofill = free - CHUNK
        # self.stream.write(SILENCE * tofill)  # Fill it with silence
        # print(free)
        return None

    def setup(self):
        """Set up the speaker for outputting audio"""
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
            channels=CHANNELS,
            rate=self.rate,
            input=False,
            output_host_api_specific_stream_info=stream_info,
            output=True,
            output_device_index=self.device_index,
        )

    def shutdown(self):
        """Close the audio stream."""
        self.stream.stop_stream()
        self.stream.close()
        self.stream = None
