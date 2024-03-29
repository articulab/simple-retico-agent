"""
python TTS/tts_test.py
"""

import time
from transformers import VitsModel, AutoTokenizer
from transformers import AutoProcessor, AutoModel
import torch
import scipy
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import soundfile as sf
from TTS.api import TTS

# text_prompt = "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."
# text_prompt = "This model is licensed under Coqui Public Model License. There's a lot that goes into a license for generative models, and you can read more of the origin story of CPML here."

text_prompts = [
    "This model is licensed under Coqui Public Model License. There's a lot that goes into a license for generative models, and you can read more of the origin story of CPML here.",
    "You are a mathematics Teacher, you must teach addition to a 8 year old child student.\
    You interact with the student through a spoken dialogue.\
    As your student is a child, you must stay gentle and supportive all the time.",
    "Child : Hello ! \n\n\
    Teacher : Hi! How are your today ? \n\n\
    Child : I am fine, and I can't wait to learn mathematics !"
]

short_prompts = [
    "Yeah !",
    "Thank you !",
    "Okay sure.",
    "No problem",
    "How is that ?",
    "What ?",
    "humm, sure yeah...",
]


#################
## Facebook TTS
model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

# inference
# inputs = tokenizer(text_prompt, return_tensors="pt")
# with torch.no_grad():
#     output = model(**inputs).waveform

# # save to wav file
# scipy.io.wavfile.write("TTS/wav_files/facebook_tts.wav", rate=model.config.sampling_rate, data=output.float().numpy().T)
# sf.write("wav_files/facebook_tts_2.wav", output.numpy(), samplerate=16000)
# sf.write("facebook_tts_2.wav", output.float().numpy(), samplerate=16000)

def generate_file(prompt, file):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform
    # save to wav file
    scipy.io.wavfile.write(file, rate=model.config.sampling_rate, data=output.float().numpy().T)

file_path_long = "TTS/wav_files/facebook_tts_long"
file_path_short = "TTS/wav_files/facebook_tts_short"

# # play sound
# # Audio(output.numpy(), rate=model.config.sampling_rate)



# #################
## Suno TTS
# from transformers import AutoProcessor, AutoModelForTextToWaveform
# processor = AutoProcessor.from_pretrained("suno/bark")
# model = AutoModelForTextToWaveform.from_pretrained("suno/bark")
# processor = AutoProcessor.from_pretrained("suno/bark")
# model = AutoModel.from_pretrained("suno/bark")

# # inference
# inputs = processor(text=text_prompt, return_tensors="pt")
# speech_values = model.generate(**inputs, do_sample=True)

# # save to wav file
# sampling_rate = model.generation_config.sample_rate
# scipy.io.wavfile.write("TTS/wav_files/bark_out.wav", rate=sampling_rate, data=speech_values.cpu().numpy().squeeze())

# play sound
# Audio(output.numpy(), rate=model.config.sampling_rate)



# #################
## MICROSOFT TTS
# processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
# model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
# vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
# # load xvector containing speaker's voice characteristics from a dataset
# # from datasets import load_dataset
# # import datasets
# # dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
# embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# # speaker_embeddings = embeddings_dataset[7306]["xvector"]
# # speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)
# # embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# # speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# # inference
# inputs = processor(text=text_prompt, return_tensors="pt")
# # speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
# speech = model.generate(inputs["input_ids"], vocoder=vocoder)

# sf.write("TTS/wav_files/microsoft_tts.wav", speech.numpy(), samplerate=16000)



# #################
## TTS API
# device = "cuda"
# # tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
# tts = TTS("tts_models/en/ek1/tacotron2", gpu=True).to(device)
# # tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC").to(device)

# # print("tts.model_name = ", tts.model_name)
# # print("tts.vocoder = ", tts.voice_converter)

# # inference & save to wav file
# # tts.tts_to_file(
# #     text=text_prompt,
# #     file_path="TTS/wav_files/tts_api_2.wav",
# #     # speaker_wav="TTS/wav_files/facebook_tts.wav",
# #     # language="en"
# #     )

# def generate_file(prompt, file):
#     tts.tts_to_file(
#         text=prompt,
#         file_path=file,
#         # speaker_wav="TTS/wav_files/facebook_tts.wav",
#         # language="en"
#         )
# file_path_long = "TTS/wav_files/tts_api_long"
# file_path_short = "TTS/wav_files/tts_api_short"


for i, p in enumerate(text_prompts): 
    start = time.time()
    generate_file(p, file_path_long+str(i)+".wav")
    print("execution_time = ", round(time.time()-start, 3))

for i, p in enumerate(short_prompts): 
    start = time.time()
    generate_file(p, file_path_short+str(i)+".wav")
    print("short execution_time = ", round(time.time()-start, 3))