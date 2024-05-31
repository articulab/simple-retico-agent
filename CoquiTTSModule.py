import csv
import datetime
from email.mime import audio
import os
import threading
import time
from hashlib import blake2b

import retico_core
import numpy as np
from TTS.api import TTS, load_config
import torch
from utils import *


class CoquiTTS:
    def __init__(
        self,
        model,
        is_multilingual,
        speaker_wav,
        language,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS(model, gpu=True).to(self.device)
        self.language = language
        self.speaker_wav = speaker_wav
        self.is_multilingual = is_multilingual

    def synthesize(self, text):
        """Takes the given text and returns the synthesized speech as 22050 Hz
        int16-encoded numpy ndarray.

        Args:
            text (str): The speech to synthesize/

        Returns:
            bytes: The speech as a 22050 Hz int16-encoded numpy ndarray
        """

        if self.is_multilingual:
            # if "multilingual" in file or "vctk" in file:
            waveforms = self.tts.tts(
                text=text,
                language=self.language,
                speaker_wav=self.speaker_wav,
            )
        else:
            waveforms = self.tts.tts(
                text=text,
            )

        # waveform = waveforms.squeeze(1).detach().numpy()[0]
        waveform = np.array(waveforms)

        # Convert float32 data [-1,1] to int16 data [-32767,32767]
        waveform = (waveform * 32767).astype(np.int16).tobytes()

        return waveform


class CoquiTTSModule(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "Coqui TTS Module"

    @staticmethod
    def description():
        return "A module that synthesizes speech using CoquiTTS."

    @staticmethod
    def input_ius():
        return [retico_core.text.TextIU]

    @staticmethod
    def output_iu():
        return retico_core.audio.AudioIU

    LANGUAGE_MAPPING = {
        "en": {
            "jenny": "tts_models/en/jenny/jenny",
            "vits": "tts_models/en/ljspeech/vits",
            "vits_neon": "tts_models/en/ljspeech/vits--neon",
            # "fast_pitch": "tts_models/en/ljspeech/fast_pitch", # bug sometimes
        },
        "multi": {
            "xtts_v2": "tts_models/multilingual/multi-dataset/xtts_v2",  # bugs
            "your_tts": "tts_models/multilingual/multi-dataset/your_tts",
        },
    }

    def __init__(
        self,
        model="jenny",
        language="en",
        speaker_wav="TTS/wav_files/tts_api/tts_models_en_jenny_jenny/long_2.wav",
        dispatch_on_finish=True,
        frame_duration=0.2,
        printing=False,
        log_file="tts.csv",
        log_folder="logs/test/16k/Recording (1)/demo",
        **kwargs
    ):
        super().__init__(**kwargs)

        # logs
        self.log_file = manage_log_folder(log_folder, log_file)

        self.printing = printing

        if language not in self.LANGUAGE_MAPPING:
            print("Unknown TTS language. Defaulting to English (en).")
            language = "en"

        if model not in self.LANGUAGE_MAPPING[language].keys():
            print(
                "Unknown model for the following TTS language : "
                + language
                + ". Defaulting to "
                + next(iter(self.LANGUAGE_MAPPING[language]))
            )
            model = next(iter(self.LANGUAGE_MAPPING[language]))

        self.tts = CoquiTTS(
            model=self.LANGUAGE_MAPPING[language][model],
            language="en",
            is_multilingual=(language == "multi"),
            speaker_wav=speaker_wav,
        )

        self.dispatch_on_finish = dispatch_on_finish
        self.frame_duration = frame_duration
        # print("\n\ntts_config audio = ", self.tts.tts.synthesizer.tts_config.get("audio"))
        # print("\nsample_rate = ", self.tts.tts.synthesizer.tts_config.get("audio")["sample_rate"])
        # print("\noutput_sample_rate = ", self.tts.tts.synthesizer.tts_config.get("audio")["output_sample_rate"])
        # print("AP = ", self.tts.tts.synthesizer.tts_model.ap)
        # print("sample rate = ", self.tts.tts.synthesizer.tts_model.ap.sample_rate)
        # self.samplerate = 48000  # samplerate of tts
        # self.samplerate = self.tts.tts.synthesizer.tts_model.ap.sample_rate
        self.samplerate = self.tts.tts.synthesizer.tts_config.get("audio")[
            "sample_rate"
        ]
        self.samplewidth = 2
        self._tts_thread_active = False
        self._latest_text = ""
        self.latest_input_iu = None
        self.audio_buffer = []
        self.audio_pointer = 0
        self.clear_after_finish = False
        self.time_logs_buffer = []

    def current_text(self):
        return "".join(iu.text for iu in self.current_input)

    def process_update(self, update_message):
        if not update_message:
            return None
        final = False
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.ADD:
                self.current_input.append(iu)
                self.latest_input_iu = iu
            elif ut == retico_core.UpdateType.REVOKE:
                self.revoke(iu)
            elif ut == retico_core.UpdateType.COMMIT:
                final = True
                # print(iu)
        current_text = self.current_text()
        # print("current_text = "+current_text)
        # print(update_message)
        if final or (
            len(current_text) - len(self._latest_text) > 15
            and not self.dispatch_on_finish
        ):
            start_time = time.time()
            start_date = datetime.datetime.now()

            self._latest_text = current_text
            chunk_size = int(self.samplerate * self.frame_duration)
            chunk_size_bytes = chunk_size * self.samplewidth
            new_audio = self.tts.synthesize(current_text)
            new_buffer = []
            i = 0
            while i < len(new_audio):
                chunk = new_audio[i : i + chunk_size_bytes]
                if len(chunk) < chunk_size_bytes:
                    chunk = chunk + b"\x00" * (chunk_size_bytes - len(chunk))
                new_buffer.append(chunk)
                i += chunk_size_bytes
            if self.clear_after_finish:
                self.audio_buffer.extend(new_buffer)
            else:
                self.audio_buffer = new_buffer

            end_time = time.time()
            end_date = datetime.datetime.now()
            if self.printing:
                print(
                    "TTS execution time = " + str(round(end_time - start_time, 3)) + "s"
                )
                print("TTS : before process ", start_date.strftime("%T.%f")[:-3])
                print("TTS : after process ", end_date.strftime("%T.%f")[:-3])
            # write_logs(
            #     self.log_file,
            #     [
            #         ["Start", start_date.strftime("%T.%f")[:-3]],
            #         ["Stop", end_date.strftime("%T.%f")[:-3]],
            #     ],
            # )
            self.time_logs_buffer.append(["Start", start_date.strftime("%T.%f")[:-3]])
            self.time_logs_buffer.append(["Stop", end_date.strftime("%T.%f")[:-3]])
        if final:
            self.clear_after_finish = True
            self.current_input = []

    def _tts_thread(self):
        t1 = time.time()
        while self._tts_thread_active:
            # this sleep time calculation is complicated and useless
            # if we don't send silence when there is no audio outputed by the tts model
            t2 = t1
            t1 = time.time()
            if t1 - t2 < self.frame_duration:
                time.sleep(self.frame_duration)
            else:
                time.sleep(max((2 * self.frame_duration) - (t1 - t2), 0))

            if self.audio_pointer >= len(self.audio_buffer):
                if self.clear_after_finish:
                    self.audio_pointer = 0
                    self.audio_buffer = []
                    self.clear_after_finish = False

                    # for WOZ : send commit when finished turn
                    iu = self.create_iu(self.latest_input_iu)
                    um = retico_core.UpdateMessage.from_iu(
                        iu, retico_core.UpdateType.COMMIT
                    )
                    self.append(um)
            else:
                raw_audio = self.audio_buffer[self.audio_pointer]
                self.audio_pointer += 1

                # Only send data to speaker when there is actual data and do not send silence ?
                iu = self.create_iu(self.latest_input_iu)
                iu.set_audio(raw_audio, 1, self.samplerate, self.samplewidth)
                um = retico_core.UpdateMessage.from_iu(iu, retico_core.UpdateType.ADD)
                self.append(um)

    def prepare_run(self):
        self.audio_pointer = 0
        self.audio_buffer = []
        self._tts_thread_active = True
        self.clear_after_finish = False
        threading.Thread(target=self._tts_thread).start()

    def shutdown(self):
        self._tts_thread_active = False
        write_logs(self.log_file, self.time_logs_buffer)
