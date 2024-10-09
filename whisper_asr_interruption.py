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

import datetime
import os
import threading
import time
import pydub
import numpy as np
import transformers
from faster_whisper import WhisperModel

import retico_core
from retico_core.utils import device_definition
from retico_core.log_utils import log_exception
from additional_IUs import DMIU

transformers.logging.set_verbosity_error()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SpeechRecognitionTurnIU(retico_core.text.SpeechRecognitionIU):
    """Same IU as SpeechRecognition, but enhanced with turn_id."""

    @staticmethod
    def type():
        return "SpeechRecognitionTurn IU"

    def __init__(self, turn_id=None, **kwargs):
        super().__init__(**kwargs)
        self.turn_id = turn_id


class WhisperASRInterruptionModule(retico_core.AbstractModule):
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
        return "WhisperASR Module"

    @staticmethod
    def description():
        return "A module that recognizes transcriptions from speech using Whisper."

    @staticmethod
    def input_ius():
        return [DMIU]

    @staticmethod
    def output_iu():
        return SpeechRecognitionTurnIU

    def __init__(
        self,
        # whisper_model="openai/whisper-base",
        # whisper_model="base.en",
        whisper_model="distil-large-v2",
        device=None,
        target_framerate=16000,
        input_framerate=16000,
        channels=1,
        sample_width=2,
        printing=False,
        **kwargs,
    ):
        """
        Initializes the WhisperASRInterruption Module.

        Args:
            whisper_model (string): name of the desired model, has to correspond to a model in the faster_whisper library.
            device (string): wether the model will be executed on cpu or gpu (using "cuda").
            language (string): language of the desired model, has to be contained in the constant LANGUAGE_MAPPING.
            speaker_wav (string): path to a wav file containing the desired voice to copy (for voice cloning models).
            target_framerate (int): model's desired audio framerate.
            input_framerate (int): framerate of the received VADTurnAudioIUs.
            channels (int): number of channels (1=mono, 2=stereo) of the received VADTurnAudioIUs.
            sample_width (int):sample width (number of bits used to encode each frame) of the received VADTurnAudioIUs.
            printing (bool, optional): You can choose to print some running info on the terminal. Defaults to False.
        """
        super().__init__(**kwargs)

        # model
        self.device = device_definition(device)
        self.model = WhisperModel(
            whisper_model, device=self.device, compute_type="int8"
        )

        # general
        self.printing = printing
        self.time_logs_buffer = []
        self._asr_thread_active = False
        self.latest_input_iu = None
        self.eos = False
        self.audio_buffer = []
        # self.vad_state = "no speaker"

        # audio
        self.target_framerate = target_framerate
        self.input_framerate = input_framerate
        self.channels = channels
        self.sample_width = sample_width
        self.start_process = True

    def resample_audio(self, audio):
        """Resample the audio's frame_rate to correspond to self.target_framerate.

        Args:
            audio (bytes): the audio received from the microphone that could need resampling.

        Returns:
            bytes: the resampled audio chunk.
        """
        if self.input_framerate != self.target_framerate:
            s = pydub.AudioSegment(
                audio,
                sample_width=self.sample_width,
                channels=self.channels,
                frame_rate=self.input_framerate,
            )
            s = s.set_frame_rate(self.target_framerate)
            return s._data
        return audio

    def add_audio(self, audio):
        """Resamples and adds the audio chunk received from the microphone to the audio buffer.

        Args:
            audio (bytes): the audio chunk received from the microphone.
        """
        audio = self.resample_audio(audio)
        self.audio_buffer.append(audio)

    def recognize(self):
        """Recreate the audio signal received by the microphone by concatenating the audio chunks
        from the audio_buffer and transcribe this concatenation into a list of predicted words.
        The function also keeps track of the user turns with the self.vad_state parameter that
        changes with the EOS recognized with the self.recognize_silence() function.

        Returns:
            (list[string], boolean): the list of words transcribed by the asr and the VAD state.
        """

        start_date = datetime.datetime.now()
        start_time = time.time()

        # faster whisper
        full_audio = b"".join(self.audio_buffer)
        audio_np = (
            np.frombuffer(full_audio, dtype=np.int16).astype(np.float32) / 32768.0
        )
        segments, _ = self.model.transcribe(audio_np)  # the segments can be streamed
        segments = list(segments)
        transcription = "".join([s.text for s in segments])

        end_date = datetime.datetime.now()
        end_time = time.time()

        if self.printing:
            print("execution time = " + str(round(end_time - start_time, 3)) + "s")
            print("ASR : before process ", start_date.strftime("%T.%f")[:-3])
            print("ASR : after process ", end_date.strftime("%T.%f")[:-3])

        return transcription

    def process_update(self, update_message):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L402

        Args:
            update_message (UpdateType): UpdateMessage that contains new IUs, if the IUs are ADD,
            they are added to the audio_buffer.
        """
        eos = False
        for iu, ut in update_message:
            if iu.action == "system_interruption":
                self.start_process = True
                continue
            # elif iu.action == "inform":
            #     if iu.vad_state:
            #         self.vad_state = iu.vad_state
            elif iu.action == "start_answer_generation":
                eos = True
                self.start_process = True
            elif iu.action == "process_audio":
                if self.input_framerate != iu.rate:
                    raise ValueError("input framerate differs from iu framerate")
                if self.start_process:
                    self.terminal_logger.info("start_process")
                    self.file_logger.info("start_process")
                    self.start_process = False
                # ADD corresponds to new audio chunks of user sentence, to generate new transcription hypothesis
                if ut == retico_core.UpdateType.ADD:
                    self.add_audio(iu.raw_audio)
                    if not self.latest_input_iu:
                        self.latest_input_iu = iu
        if eos:
            self.eos = eos

    def _asr_thread(self):
        """function used as a thread in the prepare_run function. Handles the messaging aspect of the retico module.
        Calls the WhisperASR sub-class's recognize function, and sends ADD IUs of the recognized sentence chunk to the children modules.
        If the end-of-sentence is predicted by the WhisperASR sub-class (>700ms silence), sends COMMIT IUs of the recognized full sentence.

        Using the current output to create the final prediction and COMMIT the full final transcription.
        """
        while self._asr_thread_active:
            try:
                time.sleep(0.01)
                prediction = self.recognize()
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
                            turn_id=self.latest_input_iu.turn_id,
                        )
                        self.current_output.append(output_iu)
                        um.add_iu(output_iu, retico_core.UpdateType.ADD)

                    if self.eos:
                        self.terminal_logger.info(
                            "ASR COMMIT text ius",
                            turn_id=self.current_output[-1].turn_id,
                            debug=True,
                        )
                        for iu in self.current_output:
                            self.commit(iu)
                            um.add_iu(iu, retico_core.UpdateType.COMMIT)

                        self.audio_buffer = []
                        self.current_output = []
                        self.eos = False
                        self.latest_input_iu = None

                    if len(um) != 0:
                        self.append(um)
                        if self.printing:
                            print(
                                "WHISPER SEND : ", [(iu.payload, ut) for iu, ut in um]
                            )
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
