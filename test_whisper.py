import datetime
import time
import wave
from faster_whisper import WhisperModel
import numpy as np
import pydub


def test_whisper():
    # model = WhisperModel("distil-large-v2", device="cuda", compute_type="int8")
    model = WhisperModel("base.en", device="cuda", compute_type="int8")
    wf = wave.open("audios/stereo/16k/Recording (1).wav", "rb")
    frame_rate = wf.getframerate()
    n_channels = wf.getnchannels()
    sample_width = wf.getsampwidth()
    rate = frame_rate * n_channels
    audio_data = wf.readframes(1000000)
    wf.close()

    start_time = time.time()
    start_date = datetime.datetime.now()

    if frame_rate != 16000:
        s = pydub.AudioSegment.from_wav("audios/mono/44k/Recording (1).wav")
        # s = pydub.AudioSegment(
        #     audio_data,
        #     frame_rate=frame_rate,
        #     sample_width=sample_width,
        #     n_channels=n_channels,
        # )
        s = s.set_frame_rate(16000)
        audio_data = s._data
    print("ASR : after resample ", datetime.datetime.now().strftime("%T.%f")[:-3])

    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    segments, info = model.transcribe(audio_np)  # the segments can be streamed
    segments = list(segments)
    transcription = "".join([s.text for s in segments])

    end_date = datetime.datetime.now()
    end_time = time.time()

    print("wav rate = ", frame_rate)
    print("len npa =", len(audio_np))
    print("transcription = ", transcription)
    print("execution time = " + str(round(end_time - start_time, 3)) + "s")
    print("ASR : before process ", start_date.strftime("%T.%f")[:-3])
    print("ASR : after process ", end_date.strftime("%T.%f")[:-3])


if __name__ == "__main__":
    test_whisper()
