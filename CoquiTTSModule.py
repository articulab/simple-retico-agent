from email.mime import audio
import os
import threading
import time
from hashlib import blake2b

import retico_core
import numpy as np
from TTS.api import TTS


class CoquiTTS:
    def __init__(
        self,
        model="tts_models/en/jenny/jenny",
        device="cuda",
        is_multilingual=False,
        speaker_wav="TTS/wav_files/tts_api/tts_models_en_jenny_jenny/long_2.wav",
        language="en",
    ):
        self.tts = TTS(model, gpu=True).to(device)
        self.language=language
        self.speaker_wav=speaker_wav
        self.is_multilingual=is_multilingual

    def synthesize(self, text):
        """Takes the given text and returns the synthesized speech as 22050 Hz
        int16-encoded numpy ndarray.

        Args:
            text (str): The speech to synthesize/

        Returns:
            bytes: The speech as a 22050 Hz int16-encoded numpy ndarray
        """

        if self.is_multilingual:
        # if "multilingual" in file or "vctk" in file:
            waveforms = self.tts.tts(
                text=text,
                language=self.language,
                speaker_wav=self.speaker_wav,
                )
        else:
            waveforms = self.tts.tts(
                text=text,
                )
            
        # waveform = waveforms.squeeze(1).detach().numpy()[0]
        waveform = np.array(waveforms)

        # Convert float32 data [-1,1] to int16 data [-32767,32767]
        waveform = (waveform * 32767).astype(np.int16).tobytes()

        return waveform


class CoquiTTSModule(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "Coqui TTS Module"

    @staticmethod
    def description():
        return "A module that synthesizes speech using CoquiTTS."

    @staticmethod
    def input_ius():
        return [retico_core.text.TextIU]

    @staticmethod
    def output_iu():
        return retico_core.audio.AudioIU

    LANGUAGE_MAPPING = {
        "en": {
            "jenny": "tts_models/en/jenny/jenny",
        },
        "multi": {
            "xtts_v2": "tts_models/multilingual/multi-dataset/xtts_v2",
        }
    }

    def __init__(
        self,
        model="jenny",
        language="en",
        device="cuda",
        speaker_wav=None,
        dispatch_on_finish=True,
        frame_duration=0.2,
        **kwargs
    ):
        super().__init__(**kwargs)

        if language not in self.LANGUAGE_MAPPING.keys():
            print("Unknown TTS language. Defaulting to English (en).")
            language = "en"

        self.tts = CoquiTTS(
            model=self.LANGUAGE_MAPPING[language][model],
            language=language,
            device=device,
            is_multilingual=(language=="multi"),
            speaker_wav=speaker_wav
        )

        self.dispatch_on_finish = dispatch_on_finish
        self.frame_duration = frame_duration
        # self.samplerate = 48000  # samplerate of tts
        self.samplerate = self.tts.tts.synthesizer.tts_model.ap.sample_rate
        # print("AP = ", self.tts.synthesizer.tts_model.ap)
        # print("sample rate = ", self.tts.synthesizer.tts_model.ap.sample_rate)
        # TTS.config.shared_configs.BaseAudioConfig
        self.samplewidth = 2
        self._tts_thread_active = False
        self._latest_text = ""
        self.latest_input_iu = None
        self.audio_buffer = []
        self.audio_pointer = 0
        self.clear_after_finish = False

    def current_text(self):
        return "".join(iu.text for iu in self.current_input)

    def process_update(self, update_message):
        if not update_message:
            return None
        final = False
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.ADD:
                self.current_input.append(iu)
                self.latest_input_iu = iu
            elif ut == retico_core.UpdateType.REVOKE:
                self.revoke(iu)
            elif ut == retico_core.UpdateType.COMMIT:
                final = True
                # print(iu)
        current_text = self.current_text()
        # print("current_text = "+current_text)
        # print(update_message)
        if final or (
            len(current_text) - len(self._latest_text) > 15
            and not self.dispatch_on_finish
        ):
            self._latest_text = current_text
            chunk_size = int(self.samplerate * self.frame_duration)
            chunk_size_bytes = chunk_size * self.samplewidth
            new_audio = self.tts.synthesize(current_text)
            new_buffer = []
            i = 0
            while i < len(new_audio):
                chunk = new_audio[i : i + chunk_size_bytes]
                if len(chunk) < chunk_size_bytes:
                    chunk = chunk + b"\x00" * (chunk_size_bytes - len(chunk))
                new_buffer.append(chunk)
                i += chunk_size_bytes
            if self.clear_after_finish:
                self.audio_buffer.extend(new_buffer)
            else:
                self.audio_buffer = new_buffer
        if final:
            self.clear_after_finish = True
            self.current_input = []

    def _tts_thread(self):
        t1 = time.time()
        while self._tts_thread_active:
            t2 = t1
            t1 = time.time()
            if t1 - t2 < self.frame_duration:
                time.sleep(self.frame_duration)
            else:
                time.sleep(max((2 * self.frame_duration) - (t1 - t2), 0))

            if self.audio_pointer >= len(self.audio_buffer):
                raw_audio = (
                    b"\x00"
                    * self.samplewidth
                    * int(self.samplerate * self.frame_duration)
                )
                if self.clear_after_finish:
                    self.audio_pointer = 0
                    self.audio_buffer = []
                    self.clear_after_finish = False
            else:
                raw_audio = self.audio_buffer[self.audio_pointer]
                self.audio_pointer += 1
            iu = self.create_iu(self.latest_input_iu)
            iu.set_audio(raw_audio, 1, self.samplerate, self.samplewidth)
            um = retico_core.UpdateMessage.from_iu(iu, retico_core.UpdateType.ADD)
            self.append(um)

    def prepare_run(self):
        self.audio_pointer = 0
        self.audio_buffer = []
        self._tts_thread_active = True
        self.clear_after_finish = False
        threading.Thread(target=self._tts_thread).start()

    def shutdown(self):
        self._tts_thread_active = False
