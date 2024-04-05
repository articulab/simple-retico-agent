import glob
import threading
import time
import wave
import retico_core

from retico_core.audio import AudioIU

# from audio import AudioIU, SpeechIU


# class WozMicrophoneModule(retico_core.AbstractProducingModule):
class WozMicrophoneModule(retico_core.AbstractModule):
    """A module that produces IUs containing audio signals that are captured from a wav file
    (it simulate the capture of audio by microphone with an already recorded audio saved in a wav file).
    """

    @staticmethod
    def name():
        return "WozMicrophone Module"

    @staticmethod
    def description():
        return "A producing module that produce audio from wave file."

    @staticmethod
    def output_iu():
        return AudioIU
        # return SpeechIU

    def __init__(self, folder_path="audios/", frame_length=0.02, rate=48000, **kwargs):
        """
        Initialize the Wave Module.

        Args:
            frame_length (float): The length of one frame (i.e., IU) in seconds
            rate (int): The frame rate of the recording
            sample_width (int): The width of a single sample of audio in bytes.
        """
        super().__init__(**kwargs)
        self._run_thread_active = False
        self.folder_path = folder_path
        self.r_file = None
        self.frame_length = frame_length
        self.cpt = 0
        self.tts_over = True
        self.filename_list = [f for f in glob.glob(folder_path + "*).wav")]
        print("self.filename_list = ", self.filename_list)
        self.rate = rate

    def get_file(self):
        filename = self.filename_list[self.cpt]
        print("filename = ", filename)
        self.cpt += 1
        self.r_file = wave.open(filename, "rb")
        self.n_channels = self.r_file.getnchannels()
        self.sample_width = self.r_file.getsampwidth()
        self.rate = self.r_file.getframerate() * self.n_channels
        # print("self.sample_width = ", self.sample_width)
        print("self.rate = ", self.rate)
        # print("self.n_channels = ", self.n_channels)
        # self.rate = self.r_file.getframerate() * self.sample_width ?
        # self.rate = rate
        self.chunk_size = round(self.rate * self.frame_length)

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.COMMIT:  # tts is over
                self.tts_over = True

    def _add_update_message(self):
        # print("run")
        while self._run_thread_active:
            if not self.cpt < len(self.filename_list):
                self._run_thread_active = False
            if self.tts_over:
                self.tts_over = False
                sentence_over = False
                self.get_file()
                time.sleep(2)  # 2 seconds silence before taking turn
                while not sentence_over:
                    time.sleep(
                        self.frame_length * 0.5
                    )  # Why is it not == to self.frame_length ??
                    sample = self.r_file.readframes(self.chunk_size)
                    # print("len(sample) = ", len(sample))
                    # print("self.r_file.tell() = ", self.r_file.tell())
                    sentence_over = len(sample) == 0
                    # print("sentence_over = ", sentence_over)

                    output_iu = self.create_iu()
                    output_iu.set_audio(
                        sample, self.chunk_size, self.rate, self.sample_width
                    )
                    output_iu.dispatch = True
                    um = retico_core.UpdateMessage.from_iu(
                        output_iu, retico_core.UpdateType.ADD
                    )
                    self.append(um)
                self.r_file.close()
            else:
                time.sleep(0.5)

    def prepare_run(self):
        self._run_thread_active = True
        threading.Thread(target=self._add_update_message).start()
        print("woz mic started")

    def shutdown(self):
        self._run_thread_active = False