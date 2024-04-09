"""
whisper ASR Module
==================

This module provides on-device ASR capabilities by using the whisper transformer
provided by huggingface. In addition, the ASR module provides end-of-utterance detection
(with a VAD), so that produced hypotheses are being "committed" once an utterance is
finished.
"""

import csv
import datetime
import os
import threading
import wave
import retico_core
from retico_core.audio import AudioIU
from retico_core.text import SpeechRecognitionIU
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import transformers
import pydub
import webrtcvad
import numpy as np
import time
from faster_whisper import WhisperModel

from utils import *

transformers.logging.set_verbosity_error()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class WhisperASR_2:
    def __init__(
        self,
        # whisper_model="openai/whisper-base",
        # whisper_model="base.en",
        whisper_model="distil-large-v2",
        target_framerate=16000,
        input_framerate=44100,
        channels=1,
        sample_width=2,
        silence_dur=1,
        vad_agressiveness=3,
        silence_threshold=0.75,
        language=None,
        task="transcribe",
        printing=False,
        log_file="asr.csv",
        log_folder="logs/test/16k/Recording (1)/demo",
    ):
        # self.processor = WhisperProcessor.from_pretrained(whisper_model)
        # self.model = WhisperForConditionalGeneration.from_pretrained(
        #     whisper_model
        # )

        # if language is None:
        #     self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
        #         language="english", task="transcribe"
        #     )
        #     print("Defaulting to english.")

        # else:
        #     self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
        #         language=language, task=task
        #     )
        #     print("Input Language: ", language)
        #     print("Task: ", task)
        # self.model.config.forced_decoder_ids = self.forced_decoder_ids

        self.model = WhisperModel(whisper_model, device="cuda", compute_type="int8")
        self.printing = printing

        self.audio_buffer = []
        self.target_framerate = target_framerate
        self.input_framerate = input_framerate
        self.channels = channels
        self.vad = webrtcvad.Vad(vad_agressiveness)
        self.silence_dur = silence_dur
        self.vad_state = False
        self._n_sil_frames = None
        self.silence_threshold = silence_threshold
        self.sample_width = sample_width

        self.cpt_npa = 0

        # latency logs params
        self.first_time = True
        self.first_time_stop = False
        # logs
        self.log_file = manage_log_folder(log_folder, log_file)

    def _resample_audio(self, audio):
        if self.input_framerate != self.target_framerate:
            s = pydub.AudioSegment(
                audio,
                sample_width=self.sample_width,
                channels=self.channels,
                frame_rate=self.input_framerate,
            )
            s = s.set_frame_rate(self.target_framerate)
            return s._data
        # maybe it is stereo and webrtcvad only accepts 10, 20 or 30ms mono (20ms stereo is too big)
        # if len(audio) / (sample_width * frame_rate) > 0,03 (if the audio length in more than 30ms)
        # but if stereo it's 10ms max
        # if self.get_audio_length(audio) > 0.03:
        #     half = int(len(audio) / 2)
        #     audio = audio[:half], audio[half:]
        return audio

    # def get_audio_length(self, audio):
    #     return len(audio) / (self.framerate * self.sample_width)

    def get_n_sil_frames(self):
        if not self._n_sil_frames:
            if len(self.audio_buffer) == 0:
                return None
            frame_length = len(self.audio_buffer[0]) / 2  # why divided by 2 ?
            self._n_sil_frames = int(
                self.silence_dur / (frame_length / self.target_framerate)
            )
        return self._n_sil_frames

    def recognize_silence(self):
        n_sil_frames = self.get_n_sil_frames()
        if not n_sil_frames or len(self.audio_buffer) < n_sil_frames:
            return True
        silence_counter = 0
        for a in self.audio_buffer[-n_sil_frames:]:
            if not self.vad.is_speech(a, self.target_framerate):
                silence_counter += 1
        if silence_counter >= int(self.silence_threshold * n_sil_frames):
            return True
        return False

    def add_audio(self, audio):
        audio = self._resample_audio(audio)
        self.audio_buffer.append(audio)

        # if isinstance(audio, tuple):  # or is tuple
        #     self.audio_buffer.append(audio[0])
        #     self.audio_buffer.append(audio[1])
        # else:
        #     self.audio_buffer.append(audio)

    def add_audio_2(self, audio):
        audio = self._resample_audio(audio)
        if self.vad.is_speech(audio, 16_000):
            self.audio_buffer.append(audio)

        # if type(audio) is tuple:  # or is tuple
        #     self.audio_buffer.append(audio[0])
        #     self.audio_buffer.append(audio[1])
        # else:
        #     self.audio_buffer.append(audio)

    def recognize_2(self):  # Biswesh version of asr processing in movierecommender 2022
        silence = self.recognize_silence()
        if silence:
            if len(self.audio_buffer) > 1:
                full_audio = b"".join(self.audio_buffer)
                audio_np = (
                    np.frombuffer(full_audio, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
                print("len npa =", len(audio_np))
                segments, info = self.model.transcribe(
                    audio_np
                )  # the segments can be streamed
                segments = list(segments)
                transcription = "".join([s.text for s in segments])
                self.audio_buffer = []
                return transcription, False
            return None, False
        return None, True

    def recognize(self):
        if len(self.audio_buffer) == 0:
            return None, None
        start_date = datetime.datetime.now()
        start_time = time.time()

        # print("ASR start ", datetime.datetime.now().strftime("%T.%f")[:-3])

        silence = self.recognize_silence()

        # print("ASR recog silence ", datetime.datetime.now().strftime("%T.%f")[:-3])

        if not self.vad_state and not silence:  # someone starts talking
            self.vad_state = True
            print("self.get_n_sil_frames() = ", self.get_n_sil_frames())
            self.audio_buffer = self.audio_buffer[-self.get_n_sil_frames() :]

            if self.first_time:
                self.first_time_stop = True
                self.first_time = False
                write_logs(
                    self.log_file,
                    [["Start", datetime.datetime.now().strftime("%T.%f")[:-3]]],
                )

        # print("ASR if 1 ", datetime.datetime.now().strftime("%T.%f")[:-3])

        if not self.vad_state:
            return None, False

        # print("ASR if 2 ", datetime.datetime.now().strftime("%T.%f")[:-3])

        # full_audio = b""
        # for a in self.audio_buffer:
        #     full_audio += a
        # npa = (
        #     np.frombuffer(full_audio, dtype=np.int16).astype(np.double)
        #     / 32768.0
        # )  # normalize between -1 and 1
        # if len(npa) < 10:
        #     return None, False
        # input_features = self.processor(
        #     npa, sampling_rate=16000, return_tensors="pt"
        # ).input_features
        # predicted_ids = self.model.generate(input_features)
        # transcription = self.processor.batch_decode(
        #     predicted_ids, skip_special_tokens=True
        # )[0]

        # faster whisper

        full_audio = b"".join(self.audio_buffer)

        # print("ASR join ", datetime.datetime.now().strftime("%T.%f")[:-3])

        audio_np = (
            np.frombuffer(full_audio, dtype=np.int16).astype(np.float32) / 32768.0
        )

        # print("ASR npa ", datetime.datetime.now().strftime("%T.%f")[:-3])

        print("len npa =", len(audio_np))
        self.cpt_npa += len(audio_np)
        print("cpt npa = ", self.cpt_npa)
        segments, info = self.model.transcribe(audio_np)  # the segments can be streamed

        # print("ASR transcribe ", datetime.datetime.now().strftime("%T.%f")[:-3])

        segments = list(segments)

        # print("ASR segments ", datetime.datetime.now().strftime("%T.%f")[:-3])

        transcription = "".join([s.text for s in segments])

        # print("ASR join ", datetime.datetime.now().strftime("%T.%f")[:-3])

        end_date = datetime.datetime.now()
        end_time = time.time()

        if self.printing:
            # print("ASR")
            # print("start_date ", start_date.strftime('%H:%M:%S.%MSMS'))
            # print("end_date : ", end_date.strftime('%H:%M:%S.%MSMS'))
            print("execution time = " + str(round(end_time - start_time, 3)) + "s")
            print("ASR : before process ", start_date.strftime("%T.%f")[:-3])
            print("ASR : after process ", end_date.strftime("%T.%f")[:-3])

        if silence:
            self.vad_state = False
            self.audio_buffer = []
            self.cpt_npa = 0
            print("SILENCE, emptying buffer cpt npa = ", self.cpt_npa)

            if self.first_time_stop:
                self.first_time = True
                self.first_time_stop = False
                write_logs(
                    self.log_file,
                    [["Stop", datetime.datetime.now().strftime("%T.%f")[:-3]]],
                )

        return transcription, self.vad_state

    def reset(self):
        self.vad_state = True
        self.audio_buffer = []


class WhisperASRModule_2(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "Whipser ASR Module"

    @staticmethod
    def description():
        return "A module that recognizes speech using Whisper."

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return SpeechRecognitionIU

    def __init__(
        self,
        target_framerate=16000,
        input_framerate=48000,
        silence_dur=1,
        language=None,
        task="transcribe",
        full_sentences=True,
        printing=False,
        log_file="asr.csv",
        log_folder="logs/test/16k/Recording (1)/demo",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.acr = WhisperASR_2(
            silence_dur=silence_dur,
            language=language,
            task=task,
            printing=printing,
            target_framerate=target_framerate,
            input_framerate=input_framerate,
            log_file=log_file,
            log_folder=log_folder,
        )
        self.target_framerate = target_framerate
        self.input_framerate = input_framerate
        self.silence_dur = silence_dur
        self._asr_thread_active = False
        self.latest_input_iu = None

        self.full_sentences = full_sentences
        print("full_sentences = ", full_sentences)
        if full_sentences:
            self.add_audio = self.acr.add_audio
            self.recognize = self.acr.recognize
        else:
            self.add_audio = self.acr.add_audio_2
            self.recognize = self.acr.recognize_2

    def process_update(self, update_message):
        for iu, ut in update_message:
            # Audio IUs are only added and never updated.
            if ut != retico_core.UpdateType.ADD:
                continue
            # print("IU rate = ", iu.rate)
            if self.input_framerate != iu.rate:
                raise Exception("input framerate differs from iu framerate")
                # self.input_framerate = iu.rate
                # self.acr.input_framerate = self.input_framerate
                # self.acr.input_framerate = self.input_framerate
            # Here we should check if the IU raw audio length is more than 960,
            # because webrtcvad takes max 960 (30ms for mono audio)
            # if it's not the case because it's stereo, cut the IU into two IUs ?
            # self.acr.add_audio(iu.raw_audio)
            # self.acr.add_audio_2(iu.raw_audio)
            self.add_audio(iu.raw_audio)
            if not self.latest_input_iu:
                self.latest_input_iu = iu

    def _asr_thread(self):
        while self._asr_thread_active:
            if not self.full_sentences:
                time.sleep(0.5)
            else:
                time.sleep(0.01)
            # if not self.framerate:
            #     continue
            prediction, vad = self.recognize()
            # prediction, vad = self.acr.recognize()
            # prediction, vad = self.acr.recognize_2()
            if prediction is None:
                continue
            end_of_utterance = not vad
            print("EOS = ", end_of_utterance)
            um, new_tokens = retico_core.text.get_text_increment(self, prediction)

            if len(new_tokens) == 0 and vad:
                print("Nothing new ASR, continue")
                continue

            for i, token in enumerate(new_tokens):
                output_iu = self.create_iu(self.latest_input_iu)
                eou = i == len(new_tokens) - 1 and end_of_utterance
                output_iu.set_asr_results([prediction], token, 0.0, 0.99, eou)
                self.current_output.append(output_iu)
                um.add_iu(output_iu, retico_core.UpdateType.ADD)
                # print("ADD datetime  : ", datetime.datetime.now())

            if end_of_utterance:
                print("EOS, commiting current output : ", len(self.current_output))
                print("current output = ", self.current_output)
                for iu in self.current_output:
                    self.commit(iu)
                    um.add_iu(iu, retico_core.UpdateType.COMMIT)
                    # print("COMMIT datetime  : ", datetime.datetime.now())
                self.current_output = []

            self.latest_input_iu = None
            self.append(um)

    def prepare_run(self):
        self._asr_thread_active = True
        threading.Thread(target=self._asr_thread).start()
        print("ASR started")

    def shutdown(self):
        self._asr_thread_active = False
        self.acr.reset()
