"""
coqui-ai TTS Module
==================

A retico module that provides Text-To-Speech (TTS), aligns its inputs and ouputs (text and
audio), and handles user interruption.

When receiving COMMIT TurnTextIUs, synthesizes audio (TextAlignedAudioIU) corresponding to all
IUs contained in UpdateMessage.
This module also aligns the inputed words with the outputted audio, providing the outputted
TextAlignedAudioIU with the information of the word it corresponds to (contained in the
grounded_word parameter), and its place in the agent's current sentence.
The module stops synthesizing if it receives the information that the user started talking
(user barge-in/interruption of agent turn). The interruption information is recognized by
an VADTurnAudioIU with a parameter vad_state="interruption".

This modules uses the deep learning approach implemented with coqui-ai's TTS library :
https://github.com/coqui-ai/TTS

Inputs : TurnTextIU, VADTurnAudioIU

Outputs : TextAlignedAudioIU
"""

import datetime
import threading
import time
import traceback
import retico_core
import numpy as np
import torch

from additional_IUs import *
from TTS.api import TTS
from retico_core.utils import device_definition
from retico_core.log_utils import log_exception


class CoquiTTSInterruptionModule(retico_core.AbstractModule):
    """A retico module that provides Text-To-Speech (TTS), aligns its inputs and ouputs (text and
    audio), and handles user interruption.

    When receiving COMMIT TurnTextIUs, synthesizes audio (TextAlignedAudioIU) corresponding to all
    IUs contained in UpdateMessage.
    This module also aligns the inputed words with the outputted audio, providing the outputted
    TextAlignedAudioIU with the information of the word it corresponds to (contained in the
    grounded_word parameter), and its place in the agent's current sentence.
    The module stops synthesizing if it receives the information that the user started talking
    (user barge-in/interruption of agent turn). The interruption information is recognized by
    an VADTurnAudioIU with a parameter vad_state="interruption".

    This modules uses the deep learning approach implemented with coqui-ai's TTS library :
    https://github.com/coqui-ai/TTS

    Inputs : TurnTextIU, VADTurnAudioIU

    Outputs : TextAlignedAudioIU
    """

    @staticmethod
    def name():
        return "CoquiTTS Module"

    @staticmethod
    def description():
        return (
            "A module that synthesizes speech from text using coqui-ai's TTS library."
        )

    @staticmethod
    def input_ius():
        return [TurnTextIU, VADTurnAudioIU]

    @staticmethod
    def output_iu():
        return TextAlignedAudioIU

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
        # log_file="tts.csv",
        # log_folder="logs/test/16k/Recording (1)/demo",
        device=None,
        **kwargs,
    ):
        """
        Initializes the CoquiTTSInterruption Module.

        Args:
            model (string): name of the desired model, has to be contained in the constant LANGUAGE_MAPPING.
            language (string): language of the desired model, has to be contained in the constant LANGUAGE_MAPPING.
            speaker_wav (string): path to a wav file containing the desired voice to copy (for voice cloning models).
            frame_duration (float): duration of the audio chunks contained in the outputted TextAlignedAudioIUs.
            printing (bool, optional): You can choose to print some running info on the terminal. Defaults to False.
        """
        super().__init__(**kwargs)

        # model
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

        self.model = None
        self.model_name = self.LANGUAGE_MAPPING[language][model]
        self.device = device_definition(device)
        self.language = language
        self.speaker_wav = speaker_wav
        self.is_multilingual = language == "multi"

        # audio
        self.frame_duration = frame_duration
        self.samplerate = None
        self.samplewidth = 2
        self.chunk_size = None
        self.chunk_size_bytes = None

        # general
        self.printing = printing
        # self.log_file = manage_log_folder(log_folder, log_file)
        self._tts_thread_active = False
        self.iu_buffer = []
        self.buffer_pointer = 0
        self.time_logs_buffer = []
        self.interrupted_turn = -1
        self.current_turn_id = -1

        self.first_clause = True

    def synthesize(self, text):
        """Takes the given text and returns the synthesized speech as 22050 Hz
        int16-encoded numpy ndarray.

        Args:
            text (str): The speech to synthesize

        Returns:
            bytes: The speech as a 22050 Hz int16-encoded numpy ndarray
        """

        final_outputs = self.model.tts(
            text=text,
            speaker="p225",
            return_extra_outputs=True,
            split_sentences=False,
            verbose=self.printing,
        )

        # if self.is_multilingual:
        #     # if "multilingual" in file or "vctk" in file:
        #     final_outputs = self.model.tts(
        #         text=text,
        #         language=self.language,
        #         speaker_wav=self.speaker_wav,
        #         # speaker="Ana Florence",
        #     )
        # else:
        #     final_outputs = self.model.tts(text=text, speed=1.0)

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
                # print(
                #     f"self.interrupted_turn vs iu.turn_id = {self.interrupted_turn, iu.turn_id}"
                # )
                if iu.turn_id != self.interrupted_turn:
                    if ut == retico_core.UpdateType.ADD:
                        continue
                    elif ut == retico_core.UpdateType.REVOKE:
                        self.revoke(iu)
                    elif ut == retico_core.UpdateType.COMMIT:
                        if iu.final:
                            end_of_turn = True
                        else:
                            self.current_input.append(iu)
                            end_of_clause = True
                # else:
                #     print("TTS do not process iu")
            elif isinstance(iu, VADTurnAudioIU):
                if ut == retico_core.UpdateType.ADD:
                    if iu.vad_state == "interruption":
                        self.interrupted_turn = self.current_turn_id
                        end_of_clause = False
                        end_of_turn = False
                        self.first_clause = True
                        self.current_input = []
                        self.iu_buffer = []
                        self.buffer_pointer = 0
                elif ut == retico_core.UpdateType.REVOKE:
                    continue
                elif ut == retico_core.UpdateType.COMMIT:
                    continue

        if end_of_clause:
            if self.first_clause:
                self.terminal_logger.info("start_process")
                self.file_logger.info("start_process")
                self.first_clause = False
            self.current_turn_id = self.current_input[-1].turn_id
            # print("TTS : EOC")
            start_time = time.time()
            start_date = datetime.datetime.now()

            new_buffer = self.get_new_iu_buffer_from_text_input()

            self.iu_buffer.extend(new_buffer)
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
            self.first_clause = True
            # print("TTS : EOT")
            # iu = self.create_iu()
            # iu.set_data(final=True)
            iu = self.create_iu(grounded_in=self.iu_buffer[-1].grounded_in, final=True)
            self.iu_buffer.append(iu)

        if end_of_turn and end_of_clause:
            # print("TTS : EOT & EOC")
            pass

    def get_new_iu_buffer_from_text_input(self):
        """Function that aligns the TTS inputs and outputs.
        It links the words sent by LLM to audio chunks generated by TTS model.
        As we have access to the durations of the phonems generated by the model,
        we can link the audio chunks sent to speaker to the words that it corresponds to.

        Returns:
            list[TextAlignedAudioIU]: the TextAlignedAudioIUs that will be sent to the speaker module, containing the correct informations about grounded_iu, turn_id or char_id.
        """
        # preprocess on words
        current_text, words = self.current_text_and_words()
        pre_pro_words = []
        pre_pro_words_distinct = []
        try:
            for i, w in enumerate(words):
                if w[0] == " ":
                    pre_pro_words.append(i - 1)
                    if len(pre_pro_words) >= 2:
                        pre_pro_words_distinct.append(
                            words[pre_pro_words[-2] + 1 : pre_pro_words[-1] + 1]
                        )
                    else:
                        pre_pro_words_distinct.append(words[: pre_pro_words[-1] + 1])
        except IndexError:
            print(f"INDEX ERROR : {words, pre_pro_words}")
            raise IndexError

        pre_pro_words.pop(0)
        pre_pro_words.append(len(words) - 1)
        pre_pro_words_distinct.pop(0)
        if len(pre_pro_words) >= 2:
            pre_pro_words_distinct.append(
                words[pre_pro_words[-2] + 1 : pre_pro_words[-1] + 1]
            )
        else:
            pre_pro_words_distinct.append(words[: pre_pro_words[-1] + 1])

        SPACE_TOKEN_ID = 16
        NB_FRAME_PER_DURATION = 256
        new_audio, final_outputs = self.synthesize(current_text)
        tokens = self.model.synthesizer.tts_model.tokenizer.text_to_ids(current_text)
        space_tokens_ids = []
        for i, x in enumerate(tokens):
            if x == SPACE_TOKEN_ID or i == len(tokens) - 1:
                space_tokens_ids.append(i + 1)

        # pre_tokenized_txt = [
        #     self.model.synthesizer.tts_model.tokenizer.decode([y]) for y in tokens
        # ]
        # pre_tokenized_text = [x if x != "<BLNK>" else "_" for x in pre_tokenized_txt]
        new_buffer = []
        for outputs in final_outputs:
            len_wav = len(outputs["wav"])
            durations = outputs["outputs"]["durations"].squeeze().tolist()
            # total_duration = int(sum(durations))

            wav_words_chunk_len = []
            old_len_w = 0
            for s_id in space_tokens_ids:
                wav_words_chunk_len.append(
                    int(sum(durations[old_len_w:s_id])) * NB_FRAME_PER_DURATION
                )
                # wav_words_chunk_len.append(int(sum(durations[old_len_w:s_id])) * len_wav / total_duration )
                old_len_w = s_id

            if self.printing:
                if len(pre_pro_words) > len(wav_words_chunk_len):
                    print("TTS word alignment not exact, less tokens than words")
                elif len(pre_pro_words) < len(wav_words_chunk_len):
                    print("TTS word alignment not exact, more tokens than words")

            cumsum_wav_words_chunk_len = list(np.cumsum(wav_words_chunk_len))

            i = 0
            j = 0
            while i < len_wav:
                chunk = outputs["wav"][i : i + self.chunk_size]
                # modify raw audio to match correct format
                chunk = (np.array(chunk) * 32767).astype(np.int16).tobytes()
                ## TODO : change that silence padding: padding with silence will slow down the speaker a lot
                word_id = pre_pro_words[-1]
                if len(chunk) < self.chunk_size_bytes:
                    chunk = chunk + b"\x00" * (self.chunk_size_bytes - len(chunk))
                else:
                    while i + self.chunk_size >= cumsum_wav_words_chunk_len[j]:
                        j += 1
                    if j < len(pre_pro_words):
                        word_id = pre_pro_words[j]

                temp_word = words[word_id]
                grounded_iu = self.current_input[word_id]
                words_until_word_id = words[: word_id + 1]
                len_words = [len(word) for word in words[: word_id + 1]]
                char_id = sum(len_words) - 1

                i += self.chunk_size
                # iu = self.create_iu(grounded_iu)
                # iu.set_data(
                #     audio=chunk,
                #     chunk_size=self.chunk_size,
                #     rate=self.samplerate,
                #     sample_width=self.samplewidth,
                #     grounded_word=temp_word,
                #     word_id=word_id,
                #     char_id=char_id,
                #     turn_id=grounded_iu.turn_id,
                #     clause_id=grounded_iu.clause_id,
                # )
                iu = self.create_iu(
                    grounded_in=grounded_iu,
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

        return new_buffer

    def _tts_thread(self):
        # TODO : change this function so that it sends the IUs without waiting for the IU duration to make it faster and let speaker module handle that ?
        # TODO : check if the usual system, like in the demo branch, works without this function, and having the message sending directly in process update function
        """function used as a thread in the prepare_run function. Handles the messaging aspect of the retico module. if the clear_after_finish param is True,
        it means that speech chunks have been synthesized from a sentence chunk, and the speech chunks are sent to the children modules.
        """
        t1 = time.time()
        while self._tts_thread_active:
            try:
                t2 = t1
                t1 = time.time()
                if t1 - t2 < self.frame_duration:
                    time.sleep(self.frame_duration)
                else:
                    time.sleep(max((2 * self.frame_duration) - (t1 - t2), 0))

                if self.buffer_pointer >= len(self.iu_buffer):
                    self.buffer_pointer = 0
                    self.iu_buffer = []
                else:
                    iu = self.iu_buffer[self.buffer_pointer]
                    self.buffer_pointer += 1
                    um = retico_core.UpdateMessage.from_iu(
                        iu, retico_core.UpdateType.ADD
                    )
                    self.append(um)
            except Exception as e:
                log_exception(module=self, exception=e)

    def setup(self, **kwargs):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L798
        """
        super().setup(**kwargs)
        self.model = TTS(self.model_name).to(self.device)
        self.samplerate = self.model.synthesizer.tts_config.get("audio")["sample_rate"]
        self.chunk_size = int(self.samplerate * self.frame_duration)
        self.chunk_size_bytes = self.chunk_size * self.samplewidth

    def prepare_run(self):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L808
        """
        super().prepare_run()
        self.buffer_pointer = 0
        self.iu_buffer = []
        self._tts_thread_active = True
        threading.Thread(target=self._tts_thread).start()

    def shutdown(self):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L819
        """
        super().shutdown()
        # write_logs(self.log_file, self.time_logs_buffer)
        self._tts_thread_active = False
