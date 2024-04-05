import glob
import queue
import threading
import time
import wave
import pyaudio
import retico_core

from retico_core.audio import AudioIU

# from audio import AudioIU, SpeechIU


# class WozMicrophoneModule(retico_core.AbstractProducingModule):
class WozMicrophoneModule_multiple_file(retico_core.AbstractModule):
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

    def __init__(self, folder_path="./audios/stereo/48k/", frame_length=0.02, **kwargs):
        super().__init__(**kwargs)
        self.filename_list = [f for f in glob.glob(folder_path + "*.wav")]
        print(self.filename_list)
        self.frame_length = frame_length
        self._run_thread_active = False
        self.cpt = 0
        self.tts_over = False
        self.reading_file = False

        # must be initialize in prepare run
        self.wf = None
        self.chunk_size = None

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.COMMIT:  # tts is over
                self.tts_over = True

    def _add_update_message(self):
        while self._run_thread_active:
            if self.reading_file:
                # time.sleep(self.frame_length)
                time.sleep(0.001)
                # time.sleep(0.5)
                sample = self.wf.readframes(self.chunk_size)
                # print(len(sample))
                # print(self.chunk_size)
                if len(sample) != self.chunk_size * self.n_channels * self.sample_width:
                    # print("end of audio file, not enough data")
                    sample += b"\x00" * int(
                        self.chunk_size * self.n_channels * self.sample_width
                        - len(sample)
                    )
                    # print("fixed = ", len(sample))
                    # frame = b"\x00\x00" * int(14000 * frame_duration / 1000)
                    self.reading_file = False

                output_iu = self.create_iu()

                output_iu.set_audio(
                    sample, self.chunk_size, self.rate, self.sample_width
                )
                # output_iu.dispatch = True
                um = retico_core.UpdateMessage.from_iu(
                    output_iu, retico_core.UpdateType.ADD
                )
                self.append(um)
            else:
                # if the model's answer was spoken, pick a new audio file to read
                time.sleep(1)
                if self.tts_over:
                    self.reading_file = True
                    self.tts_over = False
                    self.wf = wave.open(self.filename_list[self.cpt], "rb")
                    self.cpt += 1
                    self.n_channels = self.wf.getnchannels()
                    self.sample_width = self.wf.getsampwidth()
                    self.rate = self.wf.getframerate() * self.n_channels
                    self.chunk_size = round(self.rate * self.frame_length)
                    print("self.sample_width  = ", self.sample_width)
                    print("self.wf.getframerate()  = ", self.wf.getframerate())
                    print("rate = ", self.rate)
                    print("chunk_size = ", self.chunk_size)
                    print("cpt = ", self.cpt)

    def prepare_run(self):
        self.wf = wave.open(self.filename_list[self.cpt], "rb")
        self.reading_file = True
        self.cpt += 1
        self.n_channels = self.wf.getnchannels()
        self.sample_width = self.wf.getsampwidth()
        self.rate = self.wf.getframerate() * self.n_channels
        self.chunk_size = round(self.rate * self.frame_length)
        print("self.sample_width  = ", self.sample_width)
        print("self.wf.getframerate()  = ", self.wf.getframerate())
        print("rate = ", self.rate)
        print("chunk_size = ", self.chunk_size)
        print("cpt = ", self.cpt)
        self._run_thread_active = True
        threading.Thread(target=self._add_update_message).start()
        print("woz mic started")

    def shutdown(self):
        # self.p.terminate()
        self._run_thread_active = False
