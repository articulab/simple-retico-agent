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

        final_outputs = self.tts.tts(
            text=text,
            speaker="p225",
            return_extra_outputs=True,
        )

        # if self.is_multilingual:
        #     # if "multilingual" in file or "vctk" in file:
        #     final_outputs = self.tts.tts(
        #         text=text,
        #         language=self.language,
        #         speaker_wav=self.speaker_wav,
        #         # speaker="Ana Florence",
        #     )
        # else:
        #     final_outputs = self.tts.tts(text=text, speed=1.0)

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
        return [TurnTextIU, AudioVADIU]

    @staticmethod
    def output_iu():
        return TurnAudioIU

    LANGUAGE_MAPPING = {
        "en": {
            "jenny": "tts_models/en/jenny/jenny",
            "vits": "tts_models/en/ljspeech/vits",
            "vits_neon": "tts_models/en/ljspeech/vits--neon",
            # "fast_pitch": "tts_models/en/ljspeech/fast_pitch", # bug sometimes
            "vits_vctk": "tts_models/en/vctk/vits",
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

        print("TTS LAN = ", language)
        print("TTS MODEL = ", self.LANGUAGE_MAPPING[language][model])
        self.tts = CoquiTTSInterruption(
            model=self.LANGUAGE_MAPPING[language][model],
            language=language,
            is_multilingual=(language == "multi"),
            speaker_wav=speaker_wav,
            device=device,
        )

        self.frame_duration = frame_duration
        self.samplerate = None
        self.samplewidth = 2
        self.chunk_size = None
        self.chunk_size_bytes = None
        self._tts_thread_active = False
        self.latest_input_iu = None
        self.audio_buffer = []
        self.audio_pointer = 0
        self.time_logs_buffer = []

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
                self.audio_pointer = 0
                self.audio_buffer = []
            else:
                iu = self.audio_buffer[self.audio_pointer]
                self.audio_pointer += 1
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

        end_of_clause = False
        end_of_turn = False
        for iu, ut in update_message:
            if isinstance(iu, TurnTextIU):
                if ut == retico_core.UpdateType.ADD:
                    # self.current_input.append(iu)
                    # self.latest_input_iu = iu
                    continue
                elif ut == retico_core.UpdateType.REVOKE:
                    self.revoke(iu)
                elif ut == retico_core.UpdateType.COMMIT:
                    if iu.final:
                        end_of_turn = True
                    else:
                        self.current_input.append(iu)
                        self.latest_input_iu = iu
                        end_of_clause = True
            elif isinstance(iu, AudioVADIU):
                if ut == retico_core.UpdateType.ADD:
                    if iu.vad_state == "interruption":
                        end_of_clause = False
                        self.current_input = []
                        self.audio_buffer = []
                        self.audio_pointer = 0
                elif ut == retico_core.UpdateType.REVOKE:
                    continue
                elif ut == retico_core.UpdateType.COMMIT:
                    continue

        if end_of_clause:
            print("TTS : EOC")
            start_time = time.time()
            start_date = datetime.datetime.now()

            new_buffer = self.get_new_buffer_from_text_input()

            # ADDITION
            # new_audio, final_outputs = self.tts.synthesize(current_text)
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
            # new_buffer = self.get_new_buffer_from_text_input(chunk_size_bytes)
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

            if len(self.audio_buffer) != 0:
                self.audio_buffer.extend(new_buffer)
            else:
                self.audio_buffer = new_buffer
            self.current_input = []

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

        if end_of_turn:
            print("TTS : EOT")
            iu = self.create_iu()
            iu.set_data(final=True)
            self.audio_buffer.append(iu)

        if end_of_turn and end_of_clause:
            print("TTS : EOT & EOC")

    def get_new_buffer_from_text_input(self):
        # preprocess on words
        current_text, words = self.current_text_and_words()
        # print("current_text = ", current_text)
        # print("words = ", words)
        pre_pro_words = []
        try:
            for i, w in enumerate(words):
                if w[0] == " ":
                    pre_pro_words.append(i - 1)
        except IndexError:
            print(f"INDEX ERROR : {words, pre_pro_words}")
            raise IndexError

        pre_pro_words.pop(0)
        pre_pro_words.append(len(words) - 1)
        # print("pre_pro_words = ", pre_pro_words)

        SPACE_TOKEN_ID = 16
        NB_FRAME_PER_DURATION = 256
        PUNCTUATION = [",", ".", ":", "!", "?"]
        # print("CURRENT TEXT = ", current_text)
        new_audio, final_outputs = self.tts.synthesize(current_text)
        ## Fill buffer : the new way and with fixed duration of audio chunk
        tokens = self.tts.tts.synthesizer.tts_model.tokenizer.text_to_ids(current_text)
        pre_tokenized_txt = [
            self.tts.tts.synthesizer.tts_model.tokenizer.decode([y]) for y in tokens
        ]
        space_tokens_ids = []
        for i, x in enumerate(tokens):
            if x == SPACE_TOKEN_ID or i == len(tokens) - 1:
                space_tokens_ids.append(i + 1)
            # elif pre_tokenized_txt[i] in PUNCTUATION:
            #     space_tokens_ids.append(i)
        # space_tokens_ids = [
        #     i + 1
        #     for i, x in enumerate(tokens)
        #     if x == SPACE_TOKEN_ID or i == len(tokens) - 1
        # ]

        # replace <blnk> with space
        pre_tokenized_text = [x if x != "<BLNK>" else "_" for x in pre_tokenized_txt]
        # print("tokens = ", tokens)
        # print("space_tokens_ids = ", space_tokens_ids)
        # print("pre_tokenized_text = ", pre_tokenized_text)
        new_buffer = []
        for outputs in final_outputs:
            # intermediate parameters
            len_wav = len(outputs["wav"])
            durations = outputs["outputs"]["durations"].squeeze().tolist()
            # print("durations = ", durations)
            total_duration = int(sum(durations))

            wav_words_chunk_len = []
            old_len_w = 0
            for s_id in space_tokens_ids:
                wav_words_chunk_len.append(
                    int(sum(durations[old_len_w:s_id])) * NB_FRAME_PER_DURATION
                )
                # wav_words_chunk_len.append(int(sum(durations[old_len_w:s_id])) * len_wav / total_duration )
                old_len_w = s_id

            # TODO: Sometimes it fails because the phonems of two words are combined
            # example : "for a" -> fora in phonems
            # try to implement something that always validate this with a fusion of words in the preprocess
            print(
                f"assertion pre_pro_words, wav_words_chunk_len = {len(pre_pro_words), len(wav_words_chunk_len), pre_pro_words, wav_words_chunk_len}"
            )
            assert len(pre_pro_words) == len(wav_words_chunk_len)

            cumsum_wav_words_chunk_len = list(np.cumsum(wav_words_chunk_len))
            print("cumsum_wav_words_chunk_len = ", cumsum_wav_words_chunk_len)
            print("len_wav = ", len_wav)

            i = 0
            while i < len_wav:
                # print("i + chunk_size_bytes = ", i + chunk_size_bytes)
                chunk = outputs["wav"][i : i + self.chunk_size]
                # print(len(chunk))
                # modify raw audio to match correct format
                chunk = (np.array(chunk) * 32767).astype(np.int16).tobytes()
                ## TODO : change that silence padding: padding with silence will slow down the speaker a lot
                if len(chunk) < self.chunk_size_bytes:
                    chunk = chunk + b"\x00" * (self.chunk_size_bytes - len(chunk))
                    word_id = pre_pro_words[-1]
                    # print("if")

                else:
                    word_id = 0
                    for j, lenght in enumerate(cumsum_wav_words_chunk_len):
                        if i + self.chunk_size > lenght:
                            # print("words = ", words[j])
                            # temp_word = words[j]
                            word_id = pre_pro_words[j + 1]
                temp_word = words[word_id]
                grounded_iu = self.current_input[word_id]
                words_until_word_id = words[: word_id + 1]
                len_words = [len(word) for word in words[: word_id + 1]]
                char_id = sum(len_words) - 1
                # print("words pre = ", words_until_word_id)
                # print("words pre = ", len_words)
                # print("words pre = ", word_id)
                # print("words = ", temp_word)
                # print("len_words = ", len_words)
                # print("char_id = ", char_id)
                # print("char = ", current_text[char_id])
                # print("iu.turn_id = ", grounded_iu.turn_id)
                # print("iu.clause_id = ", grounded_iu.clause_id)

                i += self.chunk_size
                # new buffer is a IU buffer
                # new_buffer.append([chunk, temp_word, word_id, char_id])
                iu = self.create_iu(grounded_iu)
                iu.set_data(
                    audio=chunk,
                    chunk_size=self.chunk_size,
                    rate=self.samplerate,
                    sample_width=self.samplewidth,
                    grounded_word=temp_word,
                    word_id=word_id,
                    char_id=char_id,
                    turn_id=grounded_iu.turn_id,
                    clause_id=grounded_iu.clause_id,
                )
                new_buffer.append(iu)

        # This would make final the last IU of each clause
        # new_buffer[-1].final = True
        return new_buffer

    def setup(self):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L798
        """
        self.tts.setup()
        self.samplerate = self.tts.tts.synthesizer.tts_config.get("audio")[
            "sample_rate"
        ]
        self.chunk_size = int(self.samplerate * self.frame_duration)
        self.chunk_size_bytes = self.chunk_size * self.samplewidth

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
