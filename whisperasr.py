"""
whisper ASR Module
==================

This module provides on-device ASR capabilities by using the whisper transformer
provided by huggingface. In addition, the ASR module provides end-of-utterance detection
(with a VAD), so that produced hypotheses are being "committed" once an utterance is
finished.
"""

import datetime
import threading
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

transformers.logging.set_verbosity_error()


class WhisperASR:
    def __init__(
        self,
        # whisper_model="openai/whisper-base",
        # whisper_model="base.en",
        whisper_model="distil-large-v2",
        framerate=16_000,
        sample_width=2,
        silence_dur=1,
        vad_agressiveness=3,
        silence_threshold=0.75,
        language=None,
        task="transcribe",
        printing=False,
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
        self.printing=printing

        self.audio_buffer = []
        self.framerate = framerate
        self.vad = webrtcvad.Vad(vad_agressiveness)
        self.silence_dur = silence_dur
        self.vad_state = False
        self._n_sil_frames = None
        self.silence_threshold = silence_threshold
        self.sample_width = sample_width

    def _resample_audio(self, audio):
        if self.framerate != 16_000:
            # resample if framerate is not 16 kHz
            s = pydub.AudioSegment(
                audio, sample_width=2, channels=1, frame_rate=self.framerate
            )
            s = s.set_frame_rate(16_000)
            return s._data
        # maybe it is stereo and webrtcvad only accepts 10, 20 or 30ms mono (20ms stereo is too big)
        # if len(audio) / (sample_width * frame_rate) > 0,03 (if the audio length in more than 30ms)
        # but if stereo it's 10ms max
        if self.get_audio_length(audio) > 0.03:
            half = int(len(audio) / 2)
            audio = audio[:half], audio[half:]
        return audio

    def get_audio_length(self, audio):
        return len(audio) / (self.framerate * self.sample_width)

    def get_n_sil_frames(self):
        if not self._n_sil_frames:
            if len(self.audio_buffer) == 0:
                return None
            frame_length = len(self.audio_buffer[0]) / 2
            self._n_sil_frames = int(
                self.silence_dur / (frame_length / 16_000)
            )
        return self._n_sil_frames

    def recognize_silence(self):
        n_sil_frames = self.get_n_sil_frames()
        if not n_sil_frames or len(self.audio_buffer) < n_sil_frames:
            return True
        silence_counter = 0
        for a in self.audio_buffer[-n_sil_frames:]:
            if not self.vad.is_speech(a, 16_000):
                silence_counter += 1
        if silence_counter >= int(self.silence_threshold * n_sil_frames):
            return True
        return False

    def add_audio(self, audio):
        audio = self._resample_audio(audio)
        if type(audio) is tuple:  # or is tuple
            self.audio_buffer.append(audio[0])
            self.audio_buffer.append(audio[1])
        else:
            self.audio_buffer.append(audio)

    def add_audio_2(self, audio):
        audio = self._resample_audio(audio)
        if self.vad.is_speech(audio, 16_000):
            self.audio_buffer.append(audio)

        # if type(audio) is tuple:  # or is tuple
        #     self.audio_buffer.append(audio[0])
        #     self.audio_buffer.append(audio[1])
        # else:
        #     self.audio_buffer.append(audio)

    def recognize_2(self): # Biswesh version of asr processing in movierecommender 2022
        silence = self.recognize_silence()
        if silence :
            if len(self.audio_buffer) > 1:
                full_audio = b''.join(self.audio_buffer)
                audio_np = np.frombuffer(full_audio, dtype=np.int16).astype(np.float32) / 32768.0
                print("len npa =" , len(audio_np))
                segments, info = self.model.transcribe(audio_np) # the segments can be streamed
                segments = list(segments)
                transcription = ''.join([s.text for s in segments])
                self.audio_buffer = []
                return transcription, False
            return None, False
        return None, True

    def recognize(self):
        start_date = datetime.datetime.now()
        start_time = time.time()

        silence = self.recognize_silence()

        if not self.vad_state and not silence:
            self.vad_state = True
            self.audio_buffer = self.audio_buffer[-self.get_n_sil_frames():]

        if not self.vad_state:
            return None, False

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
        
        full_audio = b''.join(self.audio_buffer)
        audio_np = np.frombuffer(full_audio, dtype=np.int16).astype(np.float32) / 32768.0
        print("len npa =" , len(audio_np))
        segments, info = self.model.transcribe(audio_np) # the segments can be streamed
        segments = list(segments)
        transcription = ''.join([s.text for s in segments])

        end_date = datetime.datetime.now()
        end_time = time.time()

        if self.printing:
            # print("ASR")
            # print("start_date ", start_date.strftime('%H:%M:%S.%MSMS'))
            # print("end_date : ", end_date.strftime('%H:%M:%S.%MSMS'))
            print("execution time = " + str(round(end_time - start_time, 3)) + "s")
            print("ASR : before process ", start_date.strftime('%T.%f')[:-3])
            print("ASR : after process ", end_date.strftime('%T.%f')[:-3])

        if silence:
            self.vad_state = False
            self.audio_buffer = []

        return transcription, self.vad_state

    def reset(self):
        self.vad_state = True
        self.audio_buffer = []


class WhisperASRModule(retico_core.AbstractModule):
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
        framerate=None,
        silence_dur=1,
        language=None,
        task="transcribe",
        full_sentences=True,
        printing=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.acr = WhisperASR(
            silence_dur=silence_dur,
            language=language,
            task=task,
            printing=printing,
        )
        self.framerate = framerate
        self.silence_dur = silence_dur
        self._asr_thread_active = False
        self.latest_input_iu = None

        self.full_sentences=full_sentences
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
            if self.framerate is None:
                self.framerate = iu.rate
                self.acr.framerate = self.framerate
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
            if not self.framerate:
                continue
            prediction, vad = self.recognize()
            # prediction, vad = self.acr.recognize()
            # prediction, vad = self.acr.recognize_2()
            if prediction is None:
                continue
            end_of_utterance = not vad
            um, new_tokens = retico_core.text.get_text_increment(
                self, prediction
            )

            if len(new_tokens) == 0 and vad:
                continue

            for i, token in enumerate(new_tokens):
                output_iu = self.create_iu(self.latest_input_iu)
                eou = i == len(new_tokens) - 1 and end_of_utterance
                output_iu.set_asr_results([prediction], token, 0.0, 0.99, eou)
                self.current_output.append(output_iu)
                um.add_iu(output_iu, retico_core.UpdateType.ADD)
                # print("ADD datetime  : ", datetime.datetime.now())

            if end_of_utterance:
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
