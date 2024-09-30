"""
coqui-ai TTS Module
==================

This module provides on-device TTS capabilities by using the coqui-ai TTS library and its available
models.
"""

import datetime
from email.mime import audio
import os
import threading
import time
from hashlib import blake2b
import traceback

import retico_core
from retico_core.utils import device_definition
from retico_core.log_utils import log_exception
import numpy as np
from TTS.api import TTS, load_config
import torch
from additional_IUs import *


class CoquiTTS:
    """Sub-class of CoquiTTSModule, TTS model wrapper.
    Called with the synthesize function that generates speech (audio data) from a sentence chunk (text data).
    """

    def __init__(
        self,
        model,
        is_multilingual,
        speaker_wav,
        language,
        device=None,
    ):

        self.device = device_definition(device)
        print("self.device = TTS ", self.device)
        self.tts = None
        self.model = model
        self.language = language
        self.speaker_wav = speaker_wav
        self.is_multilingual = is_multilingual

    def setup(self, **kwargs):
        """Init chosen TTS model."""
        self.tts = TTS(self.model).to(self.device)

    def synthesize(self, text):
        """Takes the given text and returns the synthesized speech as 22050 Hz
        int16-encoded numpy ndarray.

        Args:
            text (str): The speech to synthesize

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
    """A retico module that provides Text-To-Speech (TTS) using a deep learning approach implemented with coqui-ai's TTS library : https://github.com/coqui-ai/TTS.
    This class handles the aspects related to retico architecture : messaging (update message, IUs, etc), incremental, etc.
    Has a subclass, CoquiTTS, that handles the aspects related to TTS engineering.

    Definition :
    When receiving sentence chunks from LLM, add to the current input. Start TTS synthesizing the accumulated current input when receiving a COMMIT IU
    (which, for now, is send by LLM when a ponctuation is encountered during sentence generation).
    Incremental : the TTS generation could be considered incremental because it receives sentence chunks from LLM and generates speech chunks in consequence.
    But it could be even more incremental because the TTS models could yield the generated audio data (TODO: to implement in future versions).

    Inputs : TextIU

    Outputs : AudioIU
    """

    @staticmethod
    def name():
        return "coqui-ai TTS Module"

    @staticmethod
    def description():
        return (
            "A module that synthesizes speech from text using coqui-ai's TTS library."
        )

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
        # log_file="tts.csv",
        # log_folder="logs/test/16k/Recording (1)/demo",
        device=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        # logs
        # self.log_file = manage_log_folder(log_folder, log_file)

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
            language=language,
            is_multilingual=(language == "multi"),
            speaker_wav=speaker_wav,
            device=device,
        )

        self.dispatch_on_finish = dispatch_on_finish
        self.frame_duration = frame_duration
        self.samplerate = None
        self.samplewidth = 2
        self._tts_thread_active = False
        self._latest_text = ""
        self.latest_input_iu = None
        self.audio_buffer = []
        self.audio_pointer = 0
        self.clear_after_finish = False
        self.time_logs_buffer = []

    def current_text(self):
        """Convert received IUs data accumulated in current_input list into a string.

        Returns:
            string: sentence chunk to synthesize speech from.
        """
        return "".join(iu.text for iu in self.current_input)

    def _tts_thread(self):
        """function used as a thread in the prepare_run function. Handles the messaging aspect of the retico module. if the clear_after_finish param is True,
        it means that speech chunks have been synthesized from a sentence chunk, and the speech chunks are sent to the children modules.
        """
        t1 = time.time()
        while self._tts_thread_active:
            try:
                # this sleep time calculation is complicated and useless
                # if we don't send silence when there is no audio outputted by the tts model
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
                    um = retico_core.UpdateMessage.from_iu(
                        iu, retico_core.UpdateType.ADD
                    )
                    self.append(um)
            except Exception as e:
                log_exception(module=self, exception=e)

    def process_update(self, update_message):
        """overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L402

        Args:
            update_message (UpdateType): UpdateMessage that contains new IUs to process, ADD IUs' data are added to the current_input,
            COMMIT IUs are launching the speech synthesizing (using the synthesize function) with the accumulated text data in current_input (the sentence chunk).

        Returns:
            _type_: returns None if update message is None.
        """
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
                # COMMIT IUs' data should be handled and append to the current_input ?
                final = True
        current_text = self.current_text()
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

            self.time_logs_buffer.append(["Start", start_date.strftime("%T.%f")[:-3]])
            self.time_logs_buffer.append(["Stop", end_date.strftime("%T.%f")[:-3]])

        if final:
            self.clear_after_finish = True
            self.current_input = []

    def setup(self, **kwargs):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L798
        """
        self.tts.setup()
        self.samplerate = self.tts.tts.synthesizer.tts_config.get("audio")[
            "sample_rate"
        ]

    def prepare_run(self):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L808
        """
        self.audio_pointer = 0
        self.audio_buffer = []
        self._tts_thread_active = True
        self.clear_after_finish = False
        threading.Thread(target=self._tts_thread).start()

    def shutdown(self):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L819
        """
        self._tts_thread_active = False
        # write_logs(self.log_file, self.time_logs_buffer)
