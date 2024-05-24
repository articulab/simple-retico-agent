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
import transformers
import pydub
import webrtcvad
import numpy as np
import time
from faster_whisper import WhisperModel

from utils import *

transformers.logging.set_verbosity_error()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class WhisperASR_2:
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
        silence_dur=1,
        vad_agressiveness=3,
        silence_threshold=0.75,
        printing=False,
        log_file="asr.csv",
        log_folder="logs/test/16k/Recording (1)/demo",
        device="cuda",
    ):

        self.model = WhisperModel(whisper_model, device=device, compute_type="int8")
        self.printing = printing

        self.audio_buffer = []
        self.target_framerate = target_framerate
        self.input_framerate = input_framerate
        self.channels = channels
        self.vad = webrtcvad.Vad(vad_agressiveness)
        self.silence_dur = silence_dur
        self.vad_state = False
        self._n_sil_audio_chunks = None
        self.silence_threshold = silence_threshold
        self.sample_width = sample_width

        self.cpt_npa = 0

        # latency logs params
        self.first_time = True
        self.first_time_stop = False
        self.log_file = manage_log_folder(log_folder, log_file)

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

    def get_n_sil_audio_chunks(self):
        """Returns the number of silent audio chunks needed in the audio buffer to have an EOS
        (ie. to how many audio_chunk correspond self.silence_dur)

        Returns:
            int: the number of audio chunks corresponding to the duration of self.silence_dur.
        """
        if not self._n_sil_audio_chunks:
            # nb frames in each audio chunk
            nb_frames_chunk = len(self.audio_buffer[0]) / 2
            # duration of 1 audio chunk
            dur_chunk = nb_frames_chunk / self.target_framerate
            self._n_sil_audio_chunks = int(self.silence_dur / dur_chunk)
        return self._n_sil_audio_chunks

    def recognize_silence(self):
        """Function that will calculate if the ASR consider that there is a silence long enough to be a user EOS.
        Example :
        if self.silence_threshold==0.75 (percentage) and self.silence_dur==1 (seconds),
        It returns True if, accros the frames corresponding to the last 1 second of audio, more than 75% are considered as silence by the vad.

        Returns:
            boolean : the user EOS prediction
        """

        _n_sil_audio_chunks = int(self.get_n_sil_audio_chunks())
        if not _n_sil_audio_chunks or len(self.audio_buffer) < _n_sil_audio_chunks:
            return True
        silence_counter = 0
        for a in self.audio_buffer[-_n_sil_audio_chunks:]:
            if not self.vad.is_speech(a, self.target_framerate):
                silence_counter += 1
        if silence_counter >= int(self.silence_threshold * _n_sil_audio_chunks):
            return True
        return False

    def recognize(self):
        """Recreate the audio signal received by the microphone by concatenating the audio chunks
        from the audio_buffer and transcribe this concatenation into a list of predicted words.
        The function also keeps track of the user turns with the self.vad_state parameter that changes
        with the EOS recognized with the self.recognize_silence() function.

        Returns:
            (list[string], boolean): the list of words transcribed by the asr and the VAD state.
        """
        if len(self.audio_buffer) == 0:
            return None, None

        start_date = datetime.datetime.now()
        start_time = time.time()
        silence = self.recognize_silence()

        if not self.vad_state and not silence:  # someone starts talking
            self.vad_state = True
            self.audio_buffer = self.audio_buffer[-int(self.get_n_sil_audio_chunks()) :]

            if self.first_time:
                self.first_time_stop = True
                self.first_time = False
                write_logs(
                    self.log_file,
                    [["Start", datetime.datetime.now().strftime("%T.%f")[:-3]]],
                )

        if not self.vad_state:
            return None, False

        # faster whisper
        full_audio = b"".join(self.audio_buffer)
        audio_np = (
            np.frombuffer(full_audio, dtype=np.int16).astype(np.float32) / 32768.0
        )
        self.cpt_npa += len(audio_np)
        segments, info = self.model.transcribe(audio_np)
        segments = list(segments)
        transcription = "".join([s.text for s in segments])
        end_date = datetime.datetime.now()
        end_time = time.time()

        if self.printing:
            print("execution time = " + str(round(end_time - start_time, 3)) + "s")
            print("ASR : before process ", start_date.strftime("%T.%f")[:-3])
            print("ASR : after process ", end_date.strftime("%T.%f")[:-3])

        if silence:
            self.vad_state = False
            self.audio_buffer = []
            self.cpt_npa = 0

            if self.first_time_stop:
                self.first_time = True
                self.first_time_stop = False
                write_logs(
                    self.log_file,
                    [["Stop", datetime.datetime.now().strftime("%T.%f")[:-3]]],
                )

        return transcription, self.vad_state

    def reset(self):
        """reset the vad state and empty the audio_buffer"""
        self.vad_state = True
        self.audio_buffer = []


class WhisperASRModule_2(retico_core.AbstractModule):
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
        return [AudioIU]

    @staticmethod
    def output_iu():
        return SpeechRecognitionIU

    def __init__(
        self,
        target_framerate=16000,
        input_framerate=48000,
        silence_dur=1,
        printing=False,
        log_file="asr.csv",
        log_folder="logs/test/16k/Recording (1)/demo",
        device="cuda",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.asr = WhisperASR_2(
            silence_dur=silence_dur,
            printing=printing,
            target_framerate=target_framerate,
            input_framerate=input_framerate,
            log_file=log_file,
            log_folder=log_folder,
            device=device,
        )
        self.target_framerate = target_framerate
        self.input_framerate = input_framerate
        self.silence_dur = silence_dur
        self._asr_thread_active = False
        self.latest_input_iu = None

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
            if self.input_framerate != iu.rate:
                raise Exception("input framerate differs from iu framerate")
            self.asr.add_audio(iu.raw_audio)
            if not self.latest_input_iu:
                self.latest_input_iu = iu

    def _asr_thread(self):
        while self._asr_thread_active:
            time.sleep(0.01)
            prediction, vad = self.asr.recognize()
            if prediction is None:
                continue
            end_of_utterance = not vad
            um, new_tokens = retico_core.text.get_text_increment(self, prediction)

            if len(new_tokens) == 0 and vad:
                continue

            for i, token in enumerate(new_tokens):
                output_iu = self.create_iu(self.latest_input_iu)
                eou = i == len(new_tokens) - 1 and end_of_utterance
                output_iu.set_asr_results([prediction], token, 0.0, 0.99, eou)
                self.current_output.append(output_iu)
                um.add_iu(output_iu, retico_core.UpdateType.ADD)

            if end_of_utterance:
                for iu in self.current_output:
                    self.commit(iu)
                    um.add_iu(iu, retico_core.UpdateType.COMMIT)
                self.current_output = []

            self.latest_input_iu = None
            self.append(um)

    def prepare_run(self):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L808
        """
        self._asr_thread_active = True
        threading.Thread(target=self._asr_thread).start()

    def shutdown(self):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L819
        """
        self._asr_thread_active = False
        self.asr.reset()
