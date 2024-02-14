import time
import wave
import retico_core

from retico_core.audio import AudioIU

# from audio import AudioIU, SpeechIU


class WaveModule(retico_core.AbstractProducingModule):
    """A module that produces IUs containing audio signals that are captured from a wav file
    (it simulate the capture of audio by microphone with an already recorded audio saved in a wav file).
    """

    @staticmethod
    def name():
        return "Wave Module"

    @staticmethod
    def description():
        return "A producing module that produce audio from wave file."

    @staticmethod
    def output_iu():
        return AudioIU
        # return SpeechIU

    def __init__(
        self,
        file_name="audios/test.wav",
        frame_length=0.02,
        rate=16000,
        sample_width=2,
        **kwargs
    ):
        """
        Initialize the Wave Module.

        Args:
            frame_length (float): The length of one frame (i.e., IU) in seconds
            rate (int): The frame rate of the recording
            sample_width (int): The width of a single sample of audio in bytes.
        """
        super().__init__(**kwargs)
        self.file_name = file_name
        self.r_file = None
        self.frame_length = frame_length
        self.r_file = wave.open(file_name, "rb")
        print("Number of channels", self.r_file.getnchannels())
        print("Sample width", self.r_file.getsampwidth())
        print("Frame rate.", self.r_file.getframerate())
        print("Number of frames", self.r_file.getnframes())
        print("parameters:", self.r_file.getparams())
        self.sample_width = self.r_file.getsampwidth()
        self.rate = self.r_file.getframerate() * 2
        # self.rate = self.r_file.getframerate() * self.sample_width ?
        # self.rate = rate
        self.chunk_size = round(self.rate * frame_length)
        self.r_file.close()

    def process_update(self, _):
        time.sleep(
            self.frame_length * 0.5
        )  ## Why is it not == to self.frame_length ??
        sample = self.r_file.readframes(self.chunk_size)
        output_iu = self.create_iu()
        output_iu.set_audio(
            sample, self.chunk_size, self.rate, self.sample_width
        )
        output_iu.dispatch = True
        return retico_core.UpdateMessage.from_iu(
            output_iu, retico_core.UpdateType.ADD
        )

    def setup(self):
        """Set up the wave file"""
        self.r_file = wave.open(self.file_name, "rb")

    def prepare_run(self):
        pass

    def shutdown(self):
        """Close the audio stream."""
        self.r_file.close()
