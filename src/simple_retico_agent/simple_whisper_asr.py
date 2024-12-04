"""
Simple Whisper ASR Module
=========================

A retico module that provides Automatic Speech Recognition (ASR) using a
OpenAI's Whisper model.

Once the user starts talking, periodically predicts a new transcription
hypothesis from the incremental speech received, and sends the new words
as SpeechRecognitionIUs (with UpdateType = ADD). The incorrect words
from last hypothesis are REVOKED.  When the user stops talking, predicts
a final hypothesis and sends the corresponding IUs (with UpdateType =
COMMIT).

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
from retico_core import log_utils, text
from simple_retico_agent.utils import device_definition
from simple_retico_agent.additional_IUs import VADIU

transformers.logging.set_verbosity_error()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SimpleWhisperASRModule(retico_core.AbstractModule):
    """A retico module that provides Automatic Speech Recognition (ASR) using a
    OpenAI's Whisper model.

    Once the user starts talking, periodically predicts a new
    transcription hypothesis from the incremental speech received, and
    sends the new words as SpeechRecognitionIUs (with UpdateType = ADD).
    The incorrect words from last hypothesis are REVOKED.  When the user
    stops talking, predicts a final hypothesis and sends the
    corresponding IUs (with UpdateType = COMMIT).

    The faster_whisper library is used to speed up the whisper
    inference.

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
        return text.SpeechRecognitionIU

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
        """Initializes the SimpleWhisperASRModule Module.

        Args:
            whisper_model (str, optional): name of the desired model,
                has to correspond to a model in the faster_whisper
                library. Defaults to "distil-large-v2".
            device (_type_, optional): wether the model will be executed
                on cpu or gpu (using "cuda"). Defaults to None.
            framerate (int, optional): framerate of the received VADIUs.
                Defaults to 16000.
            silence_dur (int, optional): Duration of the time interval
                over which the user's EOT will be calculated. Defaults
                to 1.
            silence_threshold (float, optional): share of IUs in the
                last silence_dur seconds to present negative VA to
                predict user EOT. Defaults to 0.75.
            bot_dur (float, optional): Duration of the time interval
                over which the user's BOT will be calculated. Defaults
                to 0.4.
            bot_threshold (float, optional): share of IUs in the last
                bot_dur seconds to present positive VA to predict user
                BOT. Defaults to 0.75.
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
        """Returns the number of audio chunks corresponding to duration. Stores
        this number in the n_chunks_param_name class argument if it hasn't been
        done before.

        Args:
            n_chunks_param_name (str): the name of class argument to
                check and/or set.
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
        """Function that predicts user BOT/EOT from VADIUs received.

        Example : if self.silence_threshold==0.75 (percentage) and
        self.bot_dur==0.4 (seconds), It predicts user BOT (returns True)
        if, across the frames corresponding to the last 400ms second of
        audio, >75% contains speech.

        Args:
            _n_audio_chunks (_type_, optional): the threshold number of
                audio chunks to recognize a user BOT or EOT. Defaults to
                None.
            threshold (float, optional): the threshold share of audio
                chunks to recognize a user BOT or EOT. Defaults to None.
            condition (Callable[], optional): function that takes an IU
                and returns a boolean, if True is returned, the
                speech_counter is incremented. Defaults to None.

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
        """Recreates the audio signal received by the microphone by
        concatenating the audio chunks from the audio_buffer and transcribes
        this concatenation into a list of predicted words.

        Returns:
            (list[string], boolean): the list of transcribed words.
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
        """Remove from current_input, the oldest IUs, that will not be
        considered to predict user BOT."""
        if len(self.current_input) > 0:
            self.current_input = self.current_input[
                -int(
                    self.get_n_audio_chunks(
                        n_chunks_param_name="_n_bot_audio_chunks", duration=self.bot_dur
                    )
                ) :
            ]

    def process_update(self, update_message):
        """Receives and stores VADIUs in the self.current_input buffer.

        Args:
            update_message (UpdateType): UpdateMessage that contains new
                IUs, if their UpdateType is ADD, they are added to the
                audio_buffer.
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
        """Function that runs on a separate thread.

        Handles the ASR prediction and IUs sending aspect of the module.
        Keeps tracks of the "vad_state" (wheter the user is currently
        speaking or not), and recognizes user BOT or EOT from VADIUs
        received. When "vad_state" == "user_speaking", predicts
        periodically new ASR hypothesis. When user EOT is recognized,
        predicts and sends a final hypothesis.
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
                log_utils.log_exception(module=self, exception=e)

    def prepare_run(self):
        super().prepare_run()
        self._asr_thread_active = True
        threading.Thread(target=self._asr_thread).start()

    def shutdown(self):
        super().shutdown()
        self._asr_thread_active = False
