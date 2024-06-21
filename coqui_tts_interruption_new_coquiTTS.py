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

import retico_core
import numpy as np
import sys

sys.path.append("C:\\Users\\Alafate\\Documents\\TTS\\")
print(sys.path)
from TTS.api import TTS, load_config
import torch
from utils import *

from utils import *
from vad_turn import AudioVADIU


class AudioTTSIU(retico_core.audio.AudioIU):
    """
    TODO: I have to decide which information I put in this IU,
    to align it with the text in the LLM module

    Attributes:
        - grounded_word (str) : The word from which the IU's raw audio is corresponding to.
    """

    @staticmethod
    def type():
        return "Audio TTS IU"

    def __init__(
        self,
        creator=None,
        iuid=0,
        previous_iu=None,
        grounded_in=None,
        audio=None,
        rate=None,
        nframes=None,
        sample_width=None,
        grounded_word=None,
        word_id=None,
        **kwargs,
    ):
        super().__init__(
            creator=creator,
            iuid=iuid,
            previous_iu=previous_iu,
            grounded_in=grounded_in,
            payload=audio,
            raw_audio=audio,
            rate=rate,
            nframes=nframes,
            sample_width=sample_width,
        )
        self.grounded_word = grounded_word
        self.word_id = word_id

    def set_grounded_word(self, grounded_word, word_id):
        """Sets the grounded_word"""
        self.grounded_word = grounded_word
        self.word_id = word_id


class CoquiTTSInterruption:
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
        self.tts = None
        self.model = model
        self.language = language
        self.speaker_wav = speaker_wav
        self.is_multilingual = is_multilingual

    def setup(self):
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
            final_outputs = self.tts.tts(
                text=text,
                language=self.language,
                speaker_wav=self.speaker_wav,
                # speaker="Ana Florence",
            )
        else:
            final_outputs = self.tts.tts(text=text, speed=1.0)

        print("FINAL OUTPUTS = ", final_outputs)
        if len(final_outputs) != 2:
            raise NotImplementedError(
                "coqui TTS should output both wavforms and outputs"
            )
        else:
            waveforms, outputs = final_outputs

        # waveform = waveforms.squeeze(1).detach().numpy()[0]
        waveform = np.array(waveforms)

        # Convert float32 data [-1,1] to int16 data [-32767,32767]
        waveform = (waveform * 32767).astype(np.int16).tobytes()

        return waveform, outputs


class CoquiTTSInterruptionModule(retico_core.AbstractModule):
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
        return [retico_core.text.TextIU, AudioVADIU]

    @staticmethod
    def output_iu():
        return AudioTTSIU

    LANGUAGE_MAPPING = {
        "en": {
            "jenny": "tts_models/en/jenny/jenny",
            "vits": "tts_models/en/ljspeech/vits",
            "vits_neon": "tts_models/en/ljspeech/vits--neon",
            "vits_vctk": "tts_models/en/vctk/vits",
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
        device=None,
        **kwargs,
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

        self.tts = CoquiTTSInterruption(
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
        self.next_IU_is_BOS = True

    def current_text(self):
        """Convert received IUs data accumulated in current_input list into a string.

        Returns:
            string: sentence chunk to synthesize speech from.
        """
        return "".join(iu.text for iu in self.current_input)

    def current_text_and_words(self):
        """Convert received IUs data accumulated in current_input list into a string.

        Returns:
            string: sentence chunk to synthesize speech from.
        """
        words = [iu.text for iu in self.current_input]
        return "".join(words), words

    def _tts_thread(self):
        # TODO : change this function so that it sends the IUs without waiting for the IU duration to make it faster and let speaker module handle that ?
        # TODO : check if the usual system, like in the demo branch, works without this function, and having the message sending directly in process update function
        """function used as a thread in the prepare_run function. Handles the messaging aspect of the retico module. if the clear_after_finish param is True,
        it means that speech chunks have been synthesized from a sentence chunk, and the speech chunks are sent to the children modules.
        """
        t1 = time.time()
        while self._tts_thread_active:
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

                    # set self.next_IU_is_BOS to True because it is the EOT and next IU sent will be a BOS
                    self.next_IU_is_BOS = True
            else:
                # Reset self._previous_iu at each beginning of turn so that speakerModule can distinguish turns
                if self.next_IU_is_BOS:
                    self._previous_iu = None
                    self.next_IU_is_BOS = False

                raw_audio, word, word_id = self.audio_buffer[self.audio_pointer]
                self.audio_pointer += 1

                # Only send data to speaker when there is actual data and do not send silence ?
                iu = self.create_iu(self.latest_input_iu)
                iu.set_audio(raw_audio, 1, self.samplerate, self.samplewidth)
                iu.set_grounded_word(word, word_id)
                um = retico_core.UpdateMessage.from_iu(iu, retico_core.UpdateType.ADD)
                self.append(um)

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
            if isinstance(iu, retico_core.text.TextIU):
                if ut == retico_core.UpdateType.ADD:
                    self.current_input.append(iu)
                    self.latest_input_iu = iu
                elif ut == retico_core.UpdateType.REVOKE:
                    # print("REVOKE : ", iu.text)
                    # print("CURRENT INPUT : ", [iu.text for iu in self.current_input])
                    self.revoke(iu)
                elif ut == retico_core.UpdateType.COMMIT:
                    # COMMIT IUs data are puncutation, should it append to current_input ?
                    # self.current_input.append(iu)
                    # self.latest_input_iu = iu
                    final = True
            elif isinstance(iu, AudioVADIU):
                if ut == retico_core.UpdateType.ADD:
                    if iu.vad_state == "interruption":
                        final = False
                        self.clear_after_finish = True
                        self.current_input = []
                        # print("TTS interruption")
                elif ut == retico_core.UpdateType.REVOKE:
                    continue
                elif ut == retico_core.UpdateType.COMMIT:
                    continue

        current_text, words = self.current_text_and_words()
        if final or (
            len(current_text) - len(self._latest_text) > 15
            and not self.dispatch_on_finish
        ):
            # print("current_text = ", current_text)
            start_time = time.time()
            start_date = datetime.datetime.now()

            self._latest_text = current_text
            chunk_size = int(self.samplerate * self.frame_duration)
            chunk_size_bytes = chunk_size * self.samplewidth

            # ADDITION
            new_audio, final_outputs = self.tts.synthesize(current_text)
            # new_audio = self.tts.synthesize(current_text)

            ## Fill buffer : the new way, but with audio chunk of the duration of the words, insteadd of a fixed duration
            # tokens = self.tts.tts.synthetizer.tts_model.tokenizer.text_to_ids(
            #     current_text
            # )
            # SPACE_TOKEN_ID = 16
            # NB_FRAME_PER_DURATION = 256
            # new_buffer = []
            # for outputs in final_outputs:
            # # intermediate parameters
            # space_tokens_ids = [
            #     i + 1
            #     for i, x in enumerate(tokens)
            #     if x == SPACE_TOKEN_ID or i == len(tokens) - 1
            # ]
            # len_wav = len(outputs["wav"])
            # durations = outputs["outputs"]["durations"].squeeze().tolist()
            # total_duration = int(sum(durations))

            # wav_words_chunk_len = []
            # old_len_w = 0
            # for s_id in space_tokens_ids:
            #     wav_words_chunk_len.append(
            #         int(sum(durations[old_len_w:s_id])) * NB_FRAME_PER_DURATION
            #     )
            #     # wav_words_chunk_len.append(int(sum(durations[old_len_w:s_id])) * len_wav / total_duration )
            #     old_len_w = s_id

            # assert len(words) == len(wav_words_chunk_len)
            # old_chunk_len = 0
            # for i, chunk_len in enumerate(wav_words_chunk_len):
            #     waveforms = outputs["wav"][
            #         old_chunk_len : old_chunk_len + int(chunk_len)
            #     ]
            #     waveforms = np.array(waveforms)
            #     waveforms = (waveforms * 32767).astype(np.int16).tobytes()
            #     old_chunk_len = old_chunk_len + int(chunk_len)
            #     # TODO: take fix chunk size of raw audio and not the word duration
            #     new_buffer.append([waveforms, words[i]])

            # ## Fill buffer : the new way and with fixed duration of audio chunk
            new_buffer = self.get_new_buffer_from_text_input(
                current_text, words, chunk_size_bytes
            )
            # tokens = self.tts.tts.synthetizer.tts_model.tokenizer.text_to_ids(
            #     current_text
            # )
            # SPACE_TOKEN_ID = 16
            # NB_FRAME_PER_DURATION = 256
            # new_buffer = []
            # i = 0
            # for outputs in final_outputs:
            #     # intermediate parameters
            #     space_tokens_ids = [
            #         i + 1
            #         for i, x in enumerate(tokens)
            #         if x == SPACE_TOKEN_ID or i == len(tokens) - 1
            #     ]
            #     len_wav = len(outputs["wav"])
            #     durations = outputs["outputs"]["durations"].squeeze().tolist()
            #     total_duration = int(sum(durations))

            #     wav_words_chunk_len = []
            #     old_len_w = 0
            #     for s_id in space_tokens_ids:
            #         wav_words_chunk_len.append(
            #             int(sum(durations[old_len_w:s_id])) * NB_FRAME_PER_DURATION
            #         )
            #         # wav_words_chunk_len.append(int(sum(durations[old_len_w:s_id])) * len_wav / total_duration )
            #         old_len_w = s_id

            #     assert len(words) == len(wav_words_chunk_len)

            #     cumsum_wav_words_chunk_len = list(np.cumsum(wav_words_chunk_len))

            #     while i < len_wav:
            #         chunk = outputs["wav"][i : i + chunk_size_bytes]
            #         if len(chunk) < chunk_size_bytes:
            #             chunk = chunk + b"\x00" * (chunk_size_bytes - len(chunk))
            #         temp_word = None
            #         for j, len in enumerate(cumsum_wav_words_chunk_len):
            #             if len > temp_word:
            #                 temp_word = words[j]
            #         new_buffer.append([chunk, temp_word])
            #         i += chunk_size_bytes

            # ## Fill buffer : old way
            # new_buffer = []
            # i = 0
            # while i < len(new_audio):
            #     chunk = new_audio[i : i + chunk_size_bytes]
            #     if len(chunk) < chunk_size_bytes:
            #         chunk = chunk + b"\x00" * (chunk_size_bytes - len(chunk))
            #     new_buffer.append(chunk)
            #     i += chunk_size_bytes

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

    def get_new_buffer_from_text_input(self, current_text, words, chunk_size_bytes):
        new_audio, final_outputs = self.tts.synthesize(current_text)
        ## Fill buffer : the new way and with fixed duration of audio chunk
        tokens = self.tts.tts.synthetizer.tts_model.tokenizer.text_to_ids(current_text)
        SPACE_TOKEN_ID = 16
        NB_FRAME_PER_DURATION = 256
        new_buffer = []
        i = 0
        for outputs in final_outputs:
            # intermediate parameters
            space_tokens_ids = [
                i + 1
                for i, x in enumerate(tokens)
                if x == SPACE_TOKEN_ID or i == len(tokens) - 1
            ]
            len_wav = len(outputs["wav"])
            durations = outputs["outputs"]["durations"].squeeze().tolist()
            total_duration = int(sum(durations))

            wav_words_chunk_len = []
            old_len_w = 0
            for s_id in space_tokens_ids:
                wav_words_chunk_len.append(
                    int(sum(durations[old_len_w:s_id])) * NB_FRAME_PER_DURATION
                )
                # wav_words_chunk_len.append(int(sum(durations[old_len_w:s_id])) * len_wav / total_duration )
                old_len_w = s_id

            assert len(words) == len(wav_words_chunk_len)

            cumsum_wav_words_chunk_len = list(np.cumsum(wav_words_chunk_len))

            while i < len_wav:
                chunk = outputs["wav"][i : i + chunk_size_bytes]
                if len(chunk) < chunk_size_bytes:
                    chunk = chunk + b"\x00" * (chunk_size_bytes - len(chunk))
                temp_word = None
                for j, len in enumerate(cumsum_wav_words_chunk_len):
                    if len > temp_word:
                        temp_word = words[j]
                new_buffer.append([chunk, temp_word, j])
                i += chunk_size_bytes
        return new_buffer

    def setup(self):
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
        write_logs(self.log_file, self.time_logs_buffer)
