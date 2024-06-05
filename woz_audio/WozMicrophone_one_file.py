import datetime
import glob
import queue
import threading
import time
import wave
import pyaudio
import retico_core
import csv
from retico_core.audio import AudioIU

from utils import *

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
        self,
        # file="audios/test/normal_mic.wav",
        file="audios/mono/16k/Recording (1).wav",
        log_file="wozmic.csv",
        log_folder="logs/test/16k/Recording (1)/demo",
        frame_length=0.02,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._run_thread_active = False
        self.file = file
        self.frame_length = frame_length

        # latency logs params
        self.first_time = True
        self.first_time_stop = True
        # logs
        self.log_file = manage_log_folder(log_folder, log_file)
        self.time_logs_buffer = []
        self.last_chunk_time = time.time()

    def _add_update_message(self):
        while self._run_thread_active:
            if self.first_time:
                self.first_time = False
                self.time_logs_buffer.append(
                    ["Start", datetime.datetime.now().strftime("%T.%f")[:-3]]
                )
            else:
                self.time_logs_buffer.append(
                    ["Stop", datetime.datetime.now().strftime("%T.%f")[:-3]]
                )

            # time.sleep(0.001)
            # time.sleep(self.frame_length)

            # waiting for frame_length time to respect the real data flow as if it was a real mic.
            # delta_time = time.time() - self.last_chunk_time
            # waiting_time = self.frame_length - delta_time
            # print("delta_time = ", delta_time)
            # print("waiting_time = ", waiting_time)
            # if waiting_time > 0:
            #     time.sleep(waiting_time)
            # self.last_chunk_time = time.time()

            if self.read_cpt < self.max_cpt:
                # sample = self.audio_buff.pop()

                sample = self.audio_data[
                    (self.chunk_size * self.sample_width)
                    * self.read_cpt : (self.chunk_size * self.sample_width)
                    * (self.read_cpt + 1)
                ]
                self.read_cpt += 1
                # sample = self.wf.readframes(self.chunk_size)
                if sample == b"":  # stop cond = file fully read
                    self._run_thread_active = False

                output_iu = self.create_iu()

                output_iu.set_audio(
                    sample, self.chunk_size, self.rate, self.sample_width
                )
                output_iu.dispatch = True
                um = retico_core.UpdateMessage.from_iu(
                    output_iu, retico_core.UpdateType.ADD
                )

                delta_time = time.time() - self.last_chunk_time
                waiting_time = self.frame_length - delta_time
                if waiting_time > 0:
                    # time.sleep(waiting_time)
                    time.sleep(0.01)
                self.append(um)
                # print("delta_time = ", delta_time)
                # print("waiting_time = ", waiting_time)
                # print("delta_time = {0:.7f}".format(delta_time))
                # print("waiting_time = {0:.7f}".format(waiting_time))
                self.last_chunk_time = time.time()

                # delta_time = time.time() - self.last_chunk_time
                # waiting_time = self.frame_length - delta_time
                # print("delta_time = ", delta_time)
                # print("waiting_time = ", waiting_time)
                # self.last_chunk_time = time.time()
                # if waiting_time > 0:
                #     time.sleep(waiting_time)

            else:  # stop cond
                # self._run_thread_active = False
                # time.sleep(0.02)
                silence_audio_chunk = b"\x00" * int(
                    self.rate * self.n_channels * self.sample_width * self.frame_length
                )
                output_iu = self.create_iu()

                output_iu.set_audio(
                    silence_audio_chunk,
                    self.chunk_size,
                    self.rate,
                    self.sample_width,
                )
                # output_iu.dispatch = True
                um = retico_core.UpdateMessage.from_iu(
                    output_iu, retico_core.UpdateType.ADD
                )
                self.append(um)

                print("stop sending")
                self._run_thread_active = False
                if self.first_time_stop:
                    self.time_logs_buffer.append(
                        ["Stop", datetime.datetime.now().strftime("%T.%f")[:-3]]
                    )
                    self.first_time_stop = False
                    write_logs(
                        self.log_file,
                        self.time_logs_buffer,
                    )
                else:
                    print("this should not happened")
                    raise NotImplementedError("2nd time entering the stop cond loop")

    def prepare_run(self):
        self.wf = wave.open(self.file, "rb")
        self.n_channels = self.wf.getnchannels()
        self.sample_width = self.wf.getsampwidth()
        self.rate = self.wf.getframerate() * self.n_channels
        self.chunk_size = round(self.rate * self.frame_length)
        self.audio_data = self.wf.readframes(1000000)
        print("len audio data = ", len(self.audio_data))
        self.read_cpt = 0
        self.max_cpt = int(len(self.audio_data) / (self.chunk_size * self.sample_width))
        print("max_cpt = ", self.max_cpt)

        # self.audio_buff = [
        #     self.audio_data[
        #         (self.chunk_size * self.sample_width)
        #         * cpt : (self.chunk_size * self.sample_width)
        #         * (cpt + 1)
        #     ]
        #     for cpt in range(self.max_cpt - 1)
        # ]

        self.wf.close()
        print("self.sample_width  = ", self.sample_width)
        print("self.wf.getframerate()  = ", self.wf.getframerate())
        print("rate = ", self.rate)
        print("chunk_size = ", self.chunk_size)

        self._run_thread_active = True
        threading.Thread(target=self._add_update_message).start()
        print("woz mic started")

    def shutdown(self):
        # self.p.terminate()
        write_logs(
            self.log_file,
            self.time_logs_buffer,
        )
        self._run_thread_active = False
