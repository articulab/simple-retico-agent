# Run the VAD on 10 ms of silence. The result should be False.
import webrtcvad

vad = webrtcvad.Vad()
sample_rate = 16000
frame_duration = 10  # ms
frame = b"\x00\x00" * int(14000 * frame_duration / 1000)
print(vad.is_speech(frame, sample_rate))
