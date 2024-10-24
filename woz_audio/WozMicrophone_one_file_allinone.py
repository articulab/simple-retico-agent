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

from additional_IUs import *
from retico_core.log_utils import *

# from audio import AudioIU, SpeechIU


# class WozMicrophoneModule(retico_core.AbstractProducingModule):
class WozMicrophoneModule_one_file_allinone(retico_core.AbstractModule):
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
        file="audios/mono/16k/Recording (8).wav",
        log_file="wozmic.csv",
        log_folder="logs/test/16k/Recording (8)/demo",
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

    def prepare_run(self):
        wf = wave.open(self.file, "rb")
        frame_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        rate = frame_rate * n_channels
        audio_data = wf.readframes(1000000)
        wf.close()
        chunk_size = round(rate * self.frame_length)
        read_cpt = 0
        max_cpt = int(len(audio_data) / (chunk_size * sample_width))
        total_time = len(audio_data) / (rate * sample_width)

        threading.Thread(
            target=self.thread,
            args=(total_time, max_cpt, audio_data, chunk_size, sample_width, rate),
        ).start()

        print("total_time = ", total_time)
        print("len audio data = ", len(audio_data))
        print("self.sample_width  = ", sample_width)
        print("self.wf.getframerate()  = ", frame_rate)
        print("rate = ", rate)
        print("chunk_size = ", chunk_size)
        print("max_cpt = ", max_cpt)
        print("woz mic started")

    def thread(self, total_time, max_cpt, audio_data, chunk_size, sample_width, rate):
        start_time = time.time()
        self.time_logs_buffer.append(
            ["Start", datetime.datetime.now().strftime("%T.%f")[:-3]]
        )
        list_ius = []
        read_cpt = 0
        while read_cpt < max_cpt:
            sample = audio_data[
                (chunk_size * sample_width)
                * read_cpt : (chunk_size * sample_width)
                * (read_cpt + 1)
            ]
            read_cpt += 1
            output_iu = self.create_iu()
            output_iu.set_audio(sample, chunk_size, rate, sample_width)
            output_iu.dispatch = True
            list_ius.append((output_iu, retico_core.UpdateType.ADD))

        # Add silence for VAD
        silence_duration = 1
        nb_silence_IUs = round(silence_duration / self.frame_length)
        print("nb_silence_IUs = ", nb_silence_IUs)
        for i in range(nb_silence_IUs):
            silence_sample = b"\x00" * chunk_size * sample_width
            output_iu = self.create_iu()
            output_iu.set_audio(
                silence_sample, rate * silence_duration, rate, sample_width
            )
            output_iu.dispatch = True
            list_ius.append((output_iu, retico_core.UpdateType.ADD))

        um = retico_core.UpdateMessage()
        um.add_ius(iu_list=list_ius)
        time.sleep(total_time + silence_duration)
        self.append(um)
        self.time_logs_buffer.append(
            ["Stop", datetime.datetime.now().strftime("%T.%f")[:-3]]
        )
        end_time = time.time()
        print("duration woz mic = ", end_time - start_time)

    def shutdown(self):
        # self.p.terminate()
        self._run_thread_active = False
        write_logs(
            self.log_file,
            self.time_logs_buffer,
        )
