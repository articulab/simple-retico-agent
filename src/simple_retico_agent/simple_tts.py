"""
Simple TTS Module
=================

A retico module that provides Text-To-Speech (TTS) to a retico system by
transforming TextFinalIUs into AudioFinalIUs, clause by clause. When
receiving COMMIT TextFinalIU from the LLM, i.e. TextFinalIUs that
consists in a full clause. Synthesizes audio (AudioFinalIUs)
corresponding to all IUs contained in UpdateMessage (the complete
clause). The module only sends TextFinalIU with a fixed raw_audio
length.

This modules uses the deep learning approach implemented with coqui-ai's
TTS library : https://github.com/coqui-ai/TTS

Inputs : TextFinalIU

Outputs : AudioFinalIU
"""

import threading
import time
import numpy as np
from TTS import api

import retico_core
from retico_core import log_utils
from simple_retico_agent.utils import device_definition
from simple_retico_agent.additional_IUs import TextFinalIU, AudioFinalIU


class SimpleTTSModule(retico_core.AbstractModule):
    """A retico module that provides Text-To-Speech (TTS) to a retico system by
    transforming TextFinalIUs into AudioFinalIUs, clause by clause. When
    receiving COMMIT TextFinalIU from the LLM, i.e. TextFinalIUs that consists
    in a full clause. Synthesizes audio (AudioFinalIUs) corresponding to all
    IUs contained in UpdateMessage (the complete clause). The module only sends
    TextFinalIU with a fixed raw_audio length.

    This modules uses the deep learning approach implemented with coqui-
    ai's TTS library : https://github.com/coqui-ai/TTS

    Inputs : TextFinalIU

    Outputs : AudioFinalIU
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
        """Initializes the SimpleTTSModule.

        Args:
            model (string): name of the desired model, has to be
                contained in the constant LANGUAGE_MAPPING.
            language (string): language of the desired model, has to be
                contained in the constant LANGUAGE_MAPPING.
            speaker_wav (string): path to a wav file containing the
                desired voice to copy (for voice cloning models).
            frame_duration (float): duration of the audio chunks
                contained in the outputted AudioFinalIUs.
            verbose (bool, optional): the verbose level of the TTS
                model. Defaults to False.
            device (string, optional): the device the module will run on
                (cuda for gpu, or cpu)
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
        """Takes the given text and synthesizes speech using the TTS model.
        Returns the synthesized speech as 22050 Hz int16-encoded numpy ndarray.

        Args:
            text (str): The text to use to synthesize speech.

        Returns:
            bytes: The speech as a 22050 Hz int16-encoded numpy ndarray.
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
        """Convert received IUs data accumulated in current_input list into a
        string.

        Returns:
            string: sentence chunk to synthesize speech from.
        """
        words = [iu.text for iu in clause_ius]
        return "".join(words), words

    def process_update(self, update_message):
        """Process the COMMIT TextFinalIUs received by appending to
        self.current_input the list of IUs corresponding to the full clause."""
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
                log_utils.log_exception(module=self, exception=e)

    def get_new_iu_buffer_from_clause_ius(self, clause_ius):
        """Function that take all TextFinalIUs from one clause, synthesizes the
        corresponding speech and split the audio into AudioFinalIUs of a fixed
        raw_audio length.

        Returns:
            list[AudioFinalIU]: the generated AudioFinalIUs, with a
                fixed raw_audio length, that will be sent to the speaker
                module.
        """
        # preprocess on words
        current_text, words = self.one_clause_text_and_words(clause_ius)
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

    def setup(self):
        super().setup()
        self.model = api.TTS(self.model_name).to(self.device)
        self.samplerate = self.model.synthesizer.tts_config.get("audio")["sample_rate"]
        self.chunk_size = int(self.samplerate * self.frame_duration)
        self.chunk_size_bytes = self.chunk_size * self.samplewidth

    def prepare_run(self):
        super().prepare_run()
        self._tts_thread_active = True
        # threading.Thread(target=self._tts_thread).start()
        threading.Thread(target=self._process_one_clause).start()

    def shutdown(self):
        super().shutdown()
        self._tts_thread_active = False
