from email.mime import audio
import os
import threading
import time
from hashlib import blake2b

import retico_core
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN
import numpy as np


class SpeechBrainTTS:
    def __init__(
        self,
        tacotron_model="speechbrain/tts-tacotron2-ljspeech",
        hifi_model="speechbrain/tts-hifigan-ljspeech",
        tmp_dir="~/.cache",
        caching=True,
    ):
        self.tacotron_model = tacotron_model
        self.hifi_model = hifi_model
        self.tmp_dir = tmp_dir.replace("~", os.path.expanduser("~"))
        self.caching = caching
        self.caching_dir = os.path.join(self.tmp_dir, "sbcache")
        if not os.path.exists(self.caching_dir):
            os.makedirs(self.caching_dir)

        self.tacotron2 = Tacotron2.from_hparams(
            source=tacotron_model, savedir=os.path.join(self.tmp_dir, "sb_tts")
        )
        self.hifi_gan = HIFIGAN.from_hparams(
            source=hifi_model, savedir=os.path.join(self.tmp_dir, "sb_vocoder")
        )

    def get_cache_path(self, text):
        """
        Creates a hash of the given TTS settings and returns a unique path to the cached version of the synthesis.
        This method does not check for the cached file to exist!

        Args:
            text (str): The text to synthesis (this is included in the hash that is used for the cache path)

        Returns (str): Path to a cached version of that synthesis.

        """
        h = blake2b(digest_size=16)
        h.update(bytes(text, "utf-8"))
        h.update(bytes(self.tacotron_model, "utf-8"))
        h.update(bytes(self.hifi_model, "utf-8"))
        text_digest = h.hexdigest()

        return os.path.join(self.caching_dir, text_digest)

    def synthesize(self, text):
        """Takes the given text and returns the synthesized speech as 22050 Hz
        int16-encoded numpy ndarray.

        Args:
            text (str): The speech to synthesize/

        Returns:
            bytes: The speech as a 22050 Hz int16-encoded numpy ndarray
        """

        cache_path = self.get_cache_path(text)
        if self.caching and os.path.isfile(cache_path):
            with open(cache_path, "rb") as cfile:
                wav_audio = cfile.read()
                return wav_audio

        # print("tacotron text = " + str(text))
        mel_output, _, _ = self.tacotron2.encode_text(text)

        # Running Vocoder (spectrogram-to-waveform)
        waveforms = self.hifi_gan.decode_batch(mel_output)

        waveform = waveforms.squeeze(1).detach().numpy()[0]

        # Convert float32 data [-1,1] to int16 data [-32767,32767]
        waveform = (waveform * 32767).astype(np.int16).tobytes()

        if self.caching:
            with open(cache_path, "wb") as cfile:
                cfile.write(waveform)

        return waveform


class SpeechBrainTTSModule(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "Speechbrain TTS Module"

    @staticmethod
    def description():
        return "A module that synthesizes speech using SpeechBrain."

    @staticmethod
    def input_ius():
        return [retico_core.text.TextIU]

    @staticmethod
    def output_iu():
        return retico_core.audio.AudioIU

    LANGUAGE_MAPPING = {
        "en": {
            "tacotron_model": "speechbrain/tts-tacotron2-ljspeech",
            "hifi_model": "speechbrain/tts-hifigan-ljspeech",
        }
    }

    def __init__(
        self, language="en", dispatch_on_finish=True, frame_duration=0.2, **kwargs
    ):
        super().__init__(**kwargs)

        if language not in self.LANGUAGE_MAPPING.keys():
            print("Unknown TTS language. Defaulting to English (en).")
            language = "en"

        self.dispatch_on_finish = dispatch_on_finish
        self.language = language
        self.tts = SpeechBrainTTS(
            tacotron_model=self.LANGUAGE_MAPPING[language]["tacotron_model"],
            hifi_model=self.LANGUAGE_MAPPING[language]["hifi_model"],
        )
        self.frame_duration = frame_duration
        self.samplerate = 22050  # samplerate of tts (fixed at 22050 for speechbrain)
        self.samplewidth = 2
        self._tts_thread_active = False
        self._latest_text = ""
        self.latest_input_iu = None
        self.audio_buffer = []
        self.audio_pointer = 0
        self.clear_after_finish = False

    def current_text(self):
        return " ".join(iu.text for iu in self.current_input)

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
