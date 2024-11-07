"""
whisper ASR Module
==================

A retico module that provides Automatic Speech Recognition (ASR) using a OpenAI's Whisper
model. Periodically predicts a new text hypothesis from the input incremental speech and
predicts a final hypothesis when it is the user end of turn.

The received VADTurnAudioIU are stored in a buffer from which a prediction is made periodically,
the words that were not present in the previous hypothesis are ADDED, in contrary, the words
that were present, but aren't anymore are REVOKED.
It recognize the user's EOT information when COMMIT VADTurnAudioIUs are received, a final
prediciton is then made and the corresponding IUs are COMMITED.

The faster_whisper library is used to speed up the whisper inference.

Inputs : VADTurnAudioIU

Outputs : SpeechRecognitionIU
"""

import os
import threading
import time
import numpy as np
import transformers
from faster_whisper import WhisperModel

import retico_core
from retico_core.text import SpeechRecognitionIU
from additional_IUs import VADIU
from retico_core.utils import device_definition
from retico_core.log_utils import log_exception

transformers.logging.set_verbosity_error()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SimpleWhisperASRModule(retico_core.AbstractModule):
    """A retico module that provides Automatic Speech Recognition (ASR) using a OpenAI's Whisper
    model. Periodically predicts a new text hypothesis from the input incremental speech and
    predicts a final hypothesis when it is the user end of turn.

    The received VADTurnAudioIU are stored in a buffer from which a prediction is made periodically,
    the words that were not present in the previous hypothesis are ADDED, in contrary, the words
    that were present, but aren't anymore are REVOKED.
    It recognize the user's EOT information when COMMIT VADTurnAudioIUs are received, a final
    prediciton is then made and the corresponding IUs are COMMITED.

    The faster_whisper library is used to speed up the whisper inference.

    Inputs : VADTurnAudioIU

    Outputs : SpeechRecognitionTurnIU
    """

    @staticmethod
    def name():
        return "ASR Whisper Simple Module"

    @staticmethod
    def description():
        return "A module that recognizes transcriptions from speech using Whisper."

    @staticmethod
    def input_ius():
        return [VADIU]

    @staticmethod
    def output_iu():
        return SpeechRecognitionIU

    def __init__(
        self,
        whisper_model="distil-large-v2",
        device=None,
        framerate=16000,
        silence_dur=1,
        silence_threshold=0.75,
        bot_dur=0.4,
        bot_threshold=0.75,
        **kwargs,
    ):
        """
        Initializes the WhisperASRInterruption Module.

        Args:
            whisper_model (string): name of the desired model, has to correspond to a model in the faster_whisper library.
            device (string): wether the model will be executed on cpu or gpu (using "cuda").
            language (string): language of the desired model, has to be contained in the constant LANGUAGE_MAPPING.
            speaker_wav (string): path to a wav file containing the desired voice to copy (for voice cloning models).
            framerate (int): framerate of the received VADTurnAudioIUs.
            channels (int): number of channels (1=mono, 2=stereo) of the received VADTurnAudioIUs.
            sample_width (int):sample width (number of bits used to encode each frame) of the received VADTurnAudioIUs.
        """
        super().__init__(**kwargs)

        # model
        self.device = device_definition(device)
        self.model = WhisperModel(
            whisper_model, device=self.device, compute_type="int8"
        )

        # general
        self._asr_thread_active = False
        self.latest_input_iu = None
        self.eos = False

        # audio
        self.framerate = framerate

        # vad
        self.silence_dur = silence_dur
        self.silence_threshold = silence_threshold
        self._n_sil_audio_chunks = None
        self.bot_dur = bot_dur
        self.bot_threshold = bot_threshold
        self._n_bot_audio_chunks = None
        self.vad_state = "user_silent"

    def get_n_audio_chunks(self, n_chunks_param_name, duration):
        """Returns the number of audio chunks corresponding to duration.
        Store this number in the n_chunks_param_name class argument if it hasn't been done before.

        Args:
            n_chunks_param_name (str): the name of class argument to check and/or set.
            duration (float): duration in second.

        Returns:
            int: the number of audio chunks corresponding to duration.
        """
        if not getattr(self, n_chunks_param_name):
            if len(self.current_input) == 0:
                return None
            first_iu = self.current_input[0]
            self.framerate = first_iu.rate
            # nb frames in each audio chunk
            nb_frames_chunk = len(first_iu.payload) / 2
            # duration of 1 audio chunk
            duration_chunk = nb_frames_chunk / self.framerate
            setattr(self, n_chunks_param_name, int(duration / duration_chunk))
        return getattr(self, n_chunks_param_name)

    def recognize_user_bot(self):
        """Return the prediction on user BOT from the current audio buffer.
        Returns True if enough audio chunks contain speech.

        Returns:
            bool: the BOT prediction.
        """
        return self.recognize_turn(
            _n_audio_chunks=self.get_n_audio_chunks(
                n_chunks_param_name="_n_bot_audio_chunks", duration=self.bot_dur
            ),
            threshold=self.bot_threshold,
            condition=lambda iu: iu.va_user,
        )

    def recognize_user_eot(self):
        """Return the prediction on user EOT from the current audio buffer.
        Returns True if enough audio chunks do not contain speech.

        Returns:
            bool: the EOT prediction.
        """
        return self.recognize_turn(
            _n_audio_chunks=self.get_n_audio_chunks(
                n_chunks_param_name="_n_sil_audio_chunks", duration=self.silence_dur
            ),
            threshold=self.silence_threshold,
            condition=lambda iu: not iu.va_user,
        )

    def recognize_agent_bot(self):
        """Return True if the last VAIU received presents a positive agent VA.

        Returns:
            bool: the BOT prediction.
        """
        return self.current_input[-1].va_agent

    def recognize_agent_eot(self):
        """Return True if the last VAIU received presents a negative agent VA.

        Returns:
            bool: the EOT prediction.
        """
        return not self.current_input[-1].va_agent

    def recognize_turn(self, _n_audio_chunks=None, threshold=None, condition=None):
        """Function that will calculate if the VAD consider that the user is talking of a long enough duration to predict a BOT.
        Example :
        if self.silence_threshold==0.75 (percentage) and self.bot_dur==0.4 (seconds),
        It returns True if, across the frames corresponding to the last 400ms second of audio, more than 75% are containing speech.

        Args:
            _n_audio_chunks (_type_, optional): the threshold number of audio chunks to recognize a user BOT or EOT. Defaults to None.
            threshold (float, optional): the threshold share of audio chunks to recognize a user BOT or EOT. Defaults to None.
            condition (Callable[], optional): function that takes an IU and returns a boolean, if True is returned, the speech_counter is incremented. Defaults to None.

        Returns:
            boolean : the user BOT or EOT prediction.
        """
        if not _n_audio_chunks or len(self.current_input) < _n_audio_chunks:
            return False
        _n_audio_chunks = int(_n_audio_chunks)
        speech_counter = sum(
            1 for iu in self.current_input[-_n_audio_chunks:] if condition(iu)
        )
        if speech_counter >= int(threshold * _n_audio_chunks):
            return True
        return False

    def recognize(self):
        """Recreate the audio signal received by the microphone by concatenating the audio chunks
        from the audio_buffer and transcribe this concatenation into a list of predicted words.
        The function also keeps track of the user turns with the self.vad_state parameter that
        changes with the EOS recognized with the self.recognize_silence() function.

        Returns:
            (list[string], boolean): the list of words transcribed by the asr and the VAD state.
        """

        # faster whisper
        full_audio = b"".join([iu.raw_audio for iu in self.current_input])
        audio_np = (
            np.frombuffer(full_audio, dtype=np.int16).astype(np.float32) / 32768.0
        )
        segments, _ = self.model.transcribe(audio_np)  # the segments can be streamed
        segments = list(segments)
        transcription = "".join([s.text for s in segments])

        return transcription

    def update_current_input(self):
        """Remove from current_input, the oldest IUs, taht will not be considered to predict user BOT."""
        if len(self.current_input) > 0:
            self.current_input = self.current_input[
                -int(
                    self.get_n_audio_chunks(
                        n_chunks_param_name="_n_bot_audio_chunks", duration=self.bot_dur
                    )
                ) :
            ]

    def process_update(self, update_message):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L402

        Args:
            update_message (UpdateType): UpdateMessage that contains new IUs, if the IUs are ADD,
            they are added to the audio_buffer.
        """
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            if self.framerate != iu.rate:
                raise Exception("input framerate differs from iu framerate")
            self.current_input.append(iu)
            if not self.latest_input_iu:
                self.latest_input_iu = iu

    def _asr_thread(self):
        """function used as a thread in the prepare_run function. Handles the messaging aspect of the retico module.
        Calls the WhisperASR sub-class's recognize function, and sends ADD IUs of the recognized sentence chunk to the children modules.
        If the end-of-sentence is predicted by the WhisperASR sub-class (>700ms silence), sends COMMIT IUs of the recognized full sentence.

        Using the current output to create the final prediction and COMMIT the full final transcription.
        """
        while self._asr_thread_active:
            try:
                time.sleep(0.01)
                if self.vad_state == "user_speaking":

                    # check for use EOT
                    user_EOT = self.recognize_user_eot()
                    self.eos = user_EOT

                    # get ASR hypothesis
                    prediction = self.recognize()
                    self.file_logger.info("predict")
                    if len(prediction) != 0:
                        um, new_tokens = retico_core.text.get_text_increment(
                            self, prediction
                        )
                        for i, token in enumerate(new_tokens):
                            output_iu = self.create_iu(
                                grounded_in=self.latest_input_iu,
                                predictions=[prediction],
                                text=token,
                                stability=0.0,
                                confidence=0.99,
                                final=self.eos and (i == (len(new_tokens) - 1)),
                            )
                            self.current_output.append(output_iu)
                            um.add_iu(output_iu, retico_core.UpdateType.ADD)

                    if user_EOT:
                        self.vad_state = "user_silent"
                        self.terminal_logger.info(
                            "ASR COMMIT text ius",
                            debug=True,
                        )
                        for iu in self.current_output:
                            self.commit(iu)
                            um.add_iu(iu, retico_core.UpdateType.COMMIT)

                        self.current_input = []
                        self.current_output = []
                        self.eos = False
                        self.latest_input_iu = None
                        self.file_logger.info("send_clause")

                    if len(um) != 0:
                        self.append(um)

                elif self.vad_state == "user_silent":
                    user_BOT = self.recognize_user_bot()
                    if user_BOT:
                        self.vad_state = "user_speaking"
                    else:
                        self.update_current_input()

            except Exception as e:
                log_exception(module=self, exception=e)

    def prepare_run(self):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L808
        """
        super().prepare_run()
        self._asr_thread_active = True
        threading.Thread(target=self._asr_thread).start()

    def shutdown(self):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L819
        """
        super().shutdown()
        self._asr_thread_active = False
