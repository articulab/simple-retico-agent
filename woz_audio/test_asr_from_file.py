import datetime
from faster_whisper import WhisperModel
import numpy as np
import wave
import pydub
import webrtcvad

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

asr = WhisperModel("distil-large-v2", device="cuda", compute_type="int8")
# asr = WhisperModel("base.en", device="cuda", compute_type="int8")
# asr = WhisperASRModule_2(printing=True, full_sentences=True, input_framerate=16000)
vad = webrtcvad.Vad(3)

f = wave.open("audios/mono/16k/Recording (6).wav", "rb")
print(f.getframerate())
data = f.readframes(1000000)
print("len data = ", len(data))
s = pydub.AudioSegment(
    data,
    sample_width=f.getsampwidth(),
    channels=f.getnchannels(),
    frame_rate=f.getframerate(),
)
s = s.set_frame_rate(16000)
data = s._data
print("len data = ", len(data))

# chunks of 0.02 sec
chunk_size = int(16000 * f.getsampwidth() * 0.02)
print("chunk_size = ", chunk_size)
print("int(len(data) / chunk_size) ", int(len(data) / chunk_size))
audio_buffer = [
    data[cpt * chunk_size : (cpt + 1) * chunk_size]
    for cpt in range(int(len(data) / chunk_size))
]
print("len audio buffer ", len(audio_buffer))
vad_buffer = [vad.is_speech(chunk, 16000) for chunk in audio_buffer]
print(vad_buffer)
cpt_last_chunks_silence = 0
for f in vad_buffer[::-1]:
    if not f:
        cpt_last_chunks_silence += 1
    else:
        break
print("the last " + str(cpt_last_chunks_silence) + " chunks are silence")
print("which is equivalent to " + str(cpt_last_chunks_silence * 0.02) + "s")


npa = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
print("len npa = ", len(npa))

print("ASR start ", datetime.datetime.now().strftime("%T.%f")[:-3])
segments, info = asr.transcribe(npa)
segments = list(segments)
transcription = "".join([s.text for s in segments])
print("ASR stop ", datetime.datetime.now().strftime("%T.%f")[:-3])

print(transcription)
