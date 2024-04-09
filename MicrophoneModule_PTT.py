import queue
import keyboard
import pyaudio
import retico_core

from retico_core.audio import MicrophoneModule


class MicrophoneModule_PTT(MicrophoneModule):

    def callback(self, in_data, frame_count, time_info, status):
        """The callback function that gets called by pyaudio.

        Args:
            in_data (bytes[]): The raw audio that is coming in from the
                microphone
            frame_count (int): The number of frames that are stored in in_data
        """
        if keyboard.is_pressed("m"):
            # print("KEY PRESSED")
            self.audio_buffer.put(in_data)
        else:
            self.audio_buffer.put(b"\x00" * self.sample_width * self.chunk_size)
        return (in_data, pyaudio.paContinue)

    def process_update(self, _):
        # print("Update")
        if not self.audio_buffer:
            # print("no buff")
            return None
        try:
            sample = self.audio_buffer.get(timeout=1.0)
        except queue.Empty:
            return None
        output_iu = self.create_iu()
        output_iu.set_audio(sample, self.chunk_size, self.rate, self.sample_width)
        return retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
