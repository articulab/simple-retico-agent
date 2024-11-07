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

import random
import threading
import time
import numpy as np
from TTS.api import TTS

import retico_core
from retico_core.utils import device_definition
from retico_core.log_utils import log_exception
from retico_core.audio import AudioIU
from simple_llm import TextFinalIU


class AudioFinalIU(AudioIU):
    """AudioIU with an additional final attribute."""

    @staticmethod
    def type():
        return "Audio Final IU"

    def __init__(self, final=False, **kwargs):
        super().__init__(
            **kwargs,
        )
        self.final = final


class SimpleTTSModule(retico_core.AbstractModule):
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
        return "TTS Simple Module"

    @staticmethod
    def description():
        return (
            "A module that synthesizes speech from text using coqui-ai's TTS library."
        )

    @staticmethod
    def input_ius():
        return [TextFinalIU]

    @staticmethod
    def output_iu():
        return AudioFinalIU

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
            "vits_vctk": "tts_models/en/vctk/vits",
        },
    }

    def __init__(
        self,
        model="jenny",
        language="en",
        speaker_wav="TTS/wav_files/tts_api/tts_models_en_jenny_jenny/long_2.wav",
        frame_duration=0.2,
        verbose=False,
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
        self.verbose = verbose
        self._tts_thread_active = False
        self.iu_buffer = []
        self.buffer_pointer = 0
        self.interrupted_turn = -1
        self.current_turn_id = -1

        self.first_clause = True
        self.space_token = None

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
            return_extra_outputs=True,
            split_sentences=False,
            verbose=self.verbose,
        )

        # if self.is_multilingual:
        #     # if "multilingual" in file or "vctk" in file:
        #     final_outputs = self.model.tts(
        #         text=text,
        #         language=self.language,
        #         speaker="p225",
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

    def one_clause_text_and_words(self, clause_ius):
        """Convert received IUs data accumulated in current_input list into a string.

        Returns:
            string: sentence chunk to synthesize speech from.
        """
        words = [iu.text for iu in clause_ius]
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

        clause_ius = []
        for iu, ut in update_message:
            if isinstance(iu, TextFinalIU):
                if ut == retico_core.UpdateType.ADD:
                    continue
                elif ut == retico_core.UpdateType.REVOKE:
                    continue
                elif ut == retico_core.UpdateType.COMMIT:
                    clause_ius.append(iu)
        if len(clause_ius) != 0:
            self.current_input.append(clause_ius)

    def _process_one_clause(self):
        while self._tts_thread_active:
            try:
                time.sleep(0.02)
                if len(self.current_input) != 0:
                    clause_ius = self.current_input.pop(0)
                    end_of_turn = clause_ius[-1].final
                    um = retico_core.UpdateMessage()
                    if end_of_turn:
                        self.terminal_logger.info(
                            "EOT TTS",
                            debug=True,
                            end_of_turn=end_of_turn,
                            clause_ius=clause_ius,
                        )
                        self.terminal_logger.info("EOT TTS")
                        self.file_logger.info("EOT")
                        self.first_clause = True
                        um.add_iu(
                            self.create_iu(grounded_in=clause_ius[-1], final=True),
                            retico_core.UpdateType.ADD,
                        )
                    else:
                        self.terminal_logger.info("EOC TTS")
                        if self.first_clause:
                            self.terminal_logger.info("start_answer_generation")
                            self.file_logger.info("start_answer_generation")
                            self.first_clause = False
                        output_ius = self.get_new_iu_buffer_from_clause_ius(clause_ius)
                        um.add_ius(
                            [(iu, retico_core.UpdateType.ADD) for iu in output_ius]
                        )
                        self.file_logger.info("send_clause")
                    self.append(um)
            except Exception as e:
                log_exception(module=self, exception=e)

    def get_new_iu_buffer_from_clause_ius(self, clause_ius):
        """Function that aligns the TTS inputs and outputs.
        It links the words sent by LLM to audio chunks generated by TTS model.
        As we have access to the durations of the phonems generated by the model,
        we can link the audio chunks sent to speaker to the words that it corresponds to.

        Returns:
            list[TextAlignedAudioIU]: the TextAlignedAudioIUs that will be sent to the speaker module, containing the correct informations about grounded_iu, turn_id or char_id.
        """
        # preprocess on words
        current_text, words = self.one_clause_text_and_words(clause_ius)
        self.terminal_logger.info(
            "TTS get iu clause", debug=True, current_text=current_text, words=words
        )
        self.file_logger.info("before_synthesize")
        new_audio, final_outputs = self.synthesize(current_text)
        self.file_logger.info("after_synthesize")

        # dispatch audio so that every IU has the same raw audio length
        new_buffer = []
        for outputs in final_outputs:
            i = 0
            while i < len(outputs["wav"]):
                chunk = outputs["wav"][i : i + self.chunk_size]
                # modify raw audio to match correct format
                chunk = (np.array(chunk) * 32767).astype(np.int16).tobytes()
                if len(chunk) <= self.chunk_size_bytes:
                    chunk = chunk + b"\x00" * (self.chunk_size_bytes - len(chunk))

                i += self.chunk_size
                iu = self.create_iu(
                    grounded_in=clause_ius[-1],
                    raw_audio=chunk,
                    chunk_size=self.chunk_size,
                    rate=self.samplerate,
                    sample_width=self.samplewidth,
                )
                new_buffer.append(iu)

        return new_buffer

    # def _tts_thread(self):
    #     """function used as a thread in the prepare_run function. Handles the messaging aspect of the retico module. if the clear_after_finish param is True,
    #     it means that speech chunks have been synthesized from a sentence chunk, and the speech chunks are sent to the children modules.
    #     """
    #     # TODO : change this function so that it sends the IUs without waiting for the IU duration to make it faster and let speaker module handle that ?
    #     # TODO : check if the usual system, like in the demo branch, works without this function, and having the message sending directly in process update function
    #     t1 = time.time()
    #     while self._tts_thread_active:
    #         try:
    #             t2 = t1
    #             t1 = time.time()
    #             if t1 - t2 < self.frame_duration:
    #                 time.sleep(self.frame_duration)
    #             else:
    #                 time.sleep(max((2 * self.frame_duration) - (t1 - t2), 0))

    #             if self.buffer_pointer >= len(self.iu_buffer):
    #                 self.buffer_pointer = 0
    #                 self.iu_buffer = []
    #             else:
    #                 iu = self.iu_buffer[self.buffer_pointer]
    #                 self.buffer_pointer += 1
    #                 um = retico_core.UpdateMessage.from_iu(
    #                     iu, retico_core.UpdateType.ADD
    #                 )
    #                 self.append(um)
    #         except Exception as e:
    #             log_exception(module=self, exception=e)

    def setup(self):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L798
        """
        super().setup()
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
        # threading.Thread(target=self._tts_thread).start()
        threading.Thread(target=self._process_one_clause).start()

    def shutdown(self):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L819
        """
        super().shutdown()
        self._tts_thread_active = False
