"""
whisper ASR Module
==================

This module provides on-device ASR capabilities by using the whisper transformer
provided by huggingface. In addition, the ASR module provides end-of-utterance detection
(with a VAD), so that produced hypotheses are being "committed" once an utterance is
finished.
"""

import datetime
import os
import threading
import retico_core
from retico_core.audio import AudioIU
from retico_core.text import SpeechRecognitionIU
import torch
import transformers
import pydub
import webrtcvad
import numpy as np
import time
from faster_whisper import WhisperModel

from utils import *
from vad_turn import AudioVADIU

transformers.logging.set_verbosity_error()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class WhisperASRInterruption:
    """Sub-class of WhisperASRModule, ASR model wrapper.
    Called with the recognize function that recognize text from speech and predicts if the recognized text corresponds to a full sentence
    (ie finishes with a silence longer than silence_threshold).
    """

    def __init__(
        self,
        # whisper_model="openai/whisper-base",
        # whisper_model="base.en",
        whisper_model="distil-large-v2",
        target_framerate=16000,
        input_framerate=44100,
        channels=1,
        sample_width=2,
        printing=False,
        log_file="asr.csv",
        log_folder="logs/test/16k/Recording (1)/demo",
        device=None,
    ):
        self.device = device_definition(device)

        self.model = WhisperModel(
            whisper_model, device=self.device, compute_type="int8"
        )
        self.printing = printing

        self.audio_buffer = []
        self.target_framerate = target_framerate
        self.input_framerate = input_framerate
        self.channels = channels
        self.sample_width = sample_width

        # latency logs params
        self.first_time = True
        self.first_time_stop = False
        # logs
        self.log_file = manage_log_folder(log_folder, log_file)
        self.time_logs_buffer = []

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
        The function also keeps track of the user turns with the self.vad_state parameter that changes
        with the EOS recognized with the self.recognize_silence() function.

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
        segments, info = self.model.transcribe(audio_np)  # the segments can be streamed
        segments = list(segments)
        transcription = "".join([s.text for s in segments])

        end_date = datetime.datetime.now()
        end_time = time.time()

        if self.printing:
            print("execution time = " + str(round(end_time - start_time, 3)) + "s")
            print("ASR : before process ", start_date.strftime("%T.%f")[:-3])
            print("ASR : after process ", end_date.strftime("%T.%f")[:-3])

        return transcription

    def reset(self):
        """reset the vad state and empty the audio_buffer"""
        # self.vad_state = True
        self.audio_buffer = []


class WhisperASRInterruptionModule(retico_core.AbstractModule):
    """A retico module that provides Automatic Speech Recognition (ASR) using a OpenAI's Whisper model.
    This class handles the aspects related to retico architecture : messaging (update message, IUs, etc), incremental, etc.
    Has a subclass, WhisperASR, that handles the aspects related to ASR engineering.

    Definition :
    When receiving audio chunks from the Microphone Module, add to the audio_buffer (using the add_audio function).
    The _asr_thread function, used as a thread in the prepare_run function, will call periodically the ASR model to recognize text from the current audio buffer.
    Alongside the recognized text, the function returns an end-of-sentence prediction, that is True if a silence longer than a fixed threshold (here, 1s) is observed.
    If an end-of-sentence is predicted, the recognized text is sent to the children modules (typically, the LLM).

    Inputs : AudioIU

    Outputs : SpeechRecognitionIU
    """

    @staticmethod
    def name():
        return "Whipser ASR Module"

    @staticmethod
    def description():
        return "A module that recognizes speech using Whisper."

    @staticmethod
    def input_ius():
        return [AudioVADIU]

    @staticmethod
    def output_iu():
        return SpeechRecognitionIU

    def __init__(
        self,
        target_framerate=16000,
        input_framerate=16000,
        printing=False,
        log_file="asr.csv",
        log_folder="logs/test/16k/Recording (1)/demo",
        device=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.asr = WhisperASRInterruption(
            printing=printing,
            target_framerate=target_framerate,
            input_framerate=input_framerate,
            log_file=log_file,
            log_folder=log_folder,
            device=device,
        )
        self.target_framerate = target_framerate
        self.input_framerate = input_framerate
        self._asr_thread_active = False
        self.latest_input_iu = None
        self.eos = False

    def process_update(self, update_message):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L402

        Args:
            update_message (UpdateType): UpdateMessage that contains new IUs, if the IUs are ADD,
            they are added to the audio_buffer.
        """
        eos = False
        for iu, ut in update_message:
            if iu.vad_state == "interruption":
                pass
            elif iu.vad_state == "user_turn":
                if self.input_framerate != iu.rate:
                    raise Exception("input framerate differs from iu framerate")
                # ADD corresponds to new audio chunks of user sentence, to generate new transcription hypothesis
                if ut == retico_core.UpdateType.ADD:
                    self.asr.add_audio(iu.raw_audio)
                    if not self.latest_input_iu:
                        self.latest_input_iu = iu
                # COMMIT corresponds to the user's full audio sentence, to generate a final transcription and send it to the LLM.
                elif ut == retico_core.UpdateType.COMMIT:
                    # self.asr.add_audio(iu.raw_audio) # already added ? if we add COMMIT IUs to audio_buffer, we'll have double audio chunks
                    # generate the final hypothesis here instead of in _asr_thead ?
                    eos = True
        self.eos = eos

    def _asr_thread(self):
        """function used as a thread in the prepare_run function. Handles the messaging aspect of the retico module.
        Calls the WhisperASR sub-class's recognize function, and sends ADD IUs of the recognized sentence chunk to the children modules.
        If the end-of-sentence is predicted by the WhisperASR sub-class (>700ms silence), sends COMMIT IUs of the recognized full sentence.

        Using the current output to create the final prediction and COMMIT the full final transcription.
        """
        # TODO: Add a REVOKE for words that were on previous hypothesis and not on the in the current one
        while self._asr_thread_active:

            time.sleep(0.01)
            prediction = self.asr.recognize()

            if len(prediction) != 0:

                um, new_tokens = retico_core.text.get_text_increment(self, prediction)
                for i, token in enumerate(new_tokens):
                    output_iu = self.create_iu(self.latest_input_iu)
                    output_iu.set_asr_results(
                        [prediction],
                        token,
                        0.0,
                        0.99,
                        self.eos and (i == (len(new_tokens) - 1)),
                    )
                    self.current_output.append(output_iu)
                    um.add_iu(output_iu, retico_core.UpdateType.ADD)

                if self.eos:
                    for iu in self.current_output:
                        self.commit(iu)
                        um.add_iu(iu, retico_core.UpdateType.COMMIT)

                    self.current_output = []
                    self.asr.reset()
                    self.eos = False

                self.latest_input_iu = None
                if len(um) != 0:
                    self.append(um)
                    # print("WHISPER SEND : ", [(iu.payload, ut) for iu, ut in um])

    def _asr_thread_2(self):
        """function used as a thread in the prepare_run function. Handles the messaging aspect of the retico module.
        Calls the WhisperASR sub-class's recognize function, and sends ADD IUs of the recognized sentence chunk to the children modules.
        If the end-of-sentence is predicted by the WhisperASR sub-class (>700ms silence), sends COMMIT IUs of the recognized full sentence.

        Having two different behaviors if EOS or not. Not using current output when EOS, directly generate IUs from last prediction.
        """
        # TODO: Add a REVOKE for words that were on previous hypothesis and not on the in the current one
        while self._asr_thread_active:
            time.sleep(0.01)

            prediction = self.asr.recognize()

            # if EOS, we'll generate the final prediction and COMMIT all words
            if self.eos:
                tokens = prediction.strip().split(" ")
                um = retico_core.UpdateMessage()
                ius = []
                for i, token in enumerate(tokens):
                    # this way will not instantiate good grounded IUs because self.latest_input_iu will be the same for every IU
                    output_iu = self.create_iu(self.latest_input_iu)
                    output_iu.set_asr_results(
                        [prediction], token, 0.0, 0.99, i == (len(tokens) - 1)
                    )
                    ius.append((output_iu, retico_core.UpdateType.COMMIT))
                    self.commit(output_iu)
                    self.current_output = []
                um.add_ius(ius)

            # if not EOS, we'll generate a new transcription hypothesis and increment the current output
            else:
                um, new_tokens = retico_core.text.get_text_increment(self, prediction)
                for i, token in enumerate(new_tokens):
                    output_iu = self.create_iu(self.latest_input_iu)
                    output_iu.set_asr_results([prediction], token, 0.0, 0.99, False)
                    self.current_output.append(output_iu)
                    um.add_iu(output_iu, retico_core.UpdateType.ADD)

            self.latest_input_iu = None
            self.append(um)

    def prepare_run(self):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L808
        """
        self._asr_thread_active = True
        threading.Thread(target=self._asr_thread).start()
        print("ASR started")

    def shutdown(self):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L819
        """
        self._asr_thread_active = False
        self.asr.reset()
