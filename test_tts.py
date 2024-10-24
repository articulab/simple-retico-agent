from TTS.api import TTS


# model = TTS("tts_models/en/vctk/vits").to("cuda")
model = TTS("tts_models/en/vctk/vits", gpu=True)

text = "My name is Marius, How are you today ? that's wonderful to hear! Alright then, let's get started with our math lesson today. Do you remember what we learned about addition in our previous class? "

for i in range(10):
    waveforms, outputs = model.tts(
        text=text,
        speaker="p225",
        return_extra_outputs=True,
        split_sentences=False,
        verbose=True,
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
