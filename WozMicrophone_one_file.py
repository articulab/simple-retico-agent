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
class WozMicrophoneModule_one_file(retico_core.AbstractModule):
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

    def __init__(
        self, file="audios/stereo/48k\Recording (1).wav", frame_length=0.02, **kwargs
    ):
        super().__init__(**kwargs)
        self._run_thread_active = False
        self.file = file
        self.frame_length = frame_length

    def _add_update_message(self):
        while self._run_thread_active:
            try:
                time.sleep(0.001)
                sample = self.wf.readframes(self.chunk_size)
                if sample == b"":  # stop cond = file fully read
                    self._run_thread_active == False

                output_iu = self.create_iu()

                output_iu.set_audio(
                    sample, self.chunk_size, self.rate, self.sample_width
                )
                output_iu.dispatch = True
                um = retico_core.UpdateMessage.from_iu(
                    output_iu, retico_core.UpdateType.ADD
                )
                self.append(um)
            except queue.Empty:
                pass

    def prepare_run(self):
        self.wf = wave.open(self.file, "rb")
        self.n_channels = self.wf.getnchannels()
        self.sample_width = self.wf.getsampwidth()
        self.rate = self.wf.getframerate() * self.n_channels
        self.chunk_size = round(self.rate * self.frame_length)
        print("self.sample_width  = ", self.sample_width)
        print("self.wf.getframerate()  = ", self.wf.getframerate())
        print("rate = ", self.rate)
        print("chunk_size = ", self.chunk_size)

        self._run_thread_active = True
        threading.Thread(target=self._add_update_message).start()
        print("woz mic started")

    def shutdown(self):
        # self.p.terminate()
        self._run_thread_active = False
