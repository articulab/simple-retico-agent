from TTS.api import TTS
import numpy as np
import scipy.io.wavfile


# model = TTS("tts_models/en/vctk/vits").to("cuda")
model = TTS("tts_models/en/vctk/vits", gpu=True).to("cuda")
frame_duration = 0.2
samplerate = model.synthesizer.tts_config.get("audio")["sample_rate"]
chunk_size = int(samplerate * frame_duration)
chunk_size_bytes = chunk_size * 2
print("samplerate = ", samplerate)
print("samplerate = ", chunk_size)
print("samplerate = ", chunk_size_bytes)

# text = "My name is Marius, How are you today ? that's wonderful to hear! Alright then, let's get started with our math lesson today. Do you remember what we learned about addition in our previous class? "

bc_text = ["Yeah !"]
# bc_text = [
#     "Yeah !",
#     "okay",
#     "alright",
#     "Yeah, okay.",
#     "uh",
#     "uh, okay",
# ]

for i, t in enumerate(bc_text):
    waveforms, outputs = model.tts(
        text=t,
        speaker="p225",
        return_extra_outputs=True,
        split_sentences=False,
        verbose=True,
    )
    print(len(waveforms))
    print(len(outputs[0]["wav"]))

    chunk = (np.array(outputs[0]["wav"]) * 32767).astype(np.int16).tobytes()
    print(len(chunk))

    # print(waveforms)
    for j, o in enumerate(outputs):
        scipy.io.wavfile.write(
            f"audios/test_tts/test_{i}_{j}.wav",
            samplerate,
            o["wav"],
        )


# model.cuda()

# model.tts_to_file(
#     text=text,
#     speaker="p225",
#     return_extra_outputs=True,
#     split_sentences=False,
#     verbose=True,
#     file_path="test_tts.wav",
# )

# (np.array(chunk) * 32767).astype(np.int16).tobytes()


# self.samplerate = self.model.synthesizer.tts_config.get("audio")["sample_rate"]
# self.chunk_size = int(self.samplerate * self.frame_duration)
# self.chunk_size_bytes = self.chunk_size * self.samplewidth
