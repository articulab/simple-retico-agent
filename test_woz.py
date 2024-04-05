import pyaudio
import wave
import sys


class AudioFile:
    # chunk = 1024
    chunk = 8000

    def __init__(self, file):
        """Init audio stream"""
        self.wf = wave.open(file, "rb")
        self.p = pyaudio.PyAudio()
        print("self.sample_width  = ", self.wf.getsampwidth())
        print("self.wf.getnchannels() = ", self.wf.getnchannels())
        print("rate = ", self.wf.getframerate())
        self.stream = self.p.open(
            format=self.p.get_format_from_width(self.wf.getsampwidth()),
            channels=self.wf.getnchannels(),
            rate=self.wf.getframerate(),
            output=True,
        )

    def play(self):
        """Play entire file"""
        data = self.wf.readframes(self.chunk)
        while data != b"":
            self.stream.write(data)
            data = self.wf.readframes(self.chunk)

    def close(self):
        """Graceful shutdown"""
        self.stream.close()
        self.p.terminate()


# Usage example for pyaudio
a = AudioFile("audios/mono/16k/Recording (1).wav")
a.play()
a.close()
