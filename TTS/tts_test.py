"""
python TTS/tts_test.py
"""

import os
import time
import numpy as np
from transformers import VitsModel, AutoTokenizer, AutoModelForTextToWaveform
from transformers import AutoProcessor, AutoModel
import torch
import scipy
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import soundfile as sf
from TTS.api import TTS, ModelManager

# text_prompt = "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."
# text_prompt = "This model is licensed under Coqui Public Model License. There's a lot that goes into a license for generative models, and you can read more of the origin story of CPML here."

text_prompts = [
    "This model is licensed under Coqui Public Model License. There's a lot that goes into a license for generative models, and you can read more of the origin story of CPML here.",
    "This model is licensed under Coqui Public Model License. There's a lot that goes into a license for generative models, and you can read more of the origin story of CPML here.",
    "You are a mathematics Teacher, you must teach addition to a 8 year old child student.\
    You interact with the student through a spoken dialogue.\
    As your student is a child, you must stay gentle and supportive all the time.",
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
# ## Facebook TTS
# # tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
# # model = VitsModel.from_pretrained("facebook/mms-tts-eng")

# tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
# model = AutoModelForTextToWaveform.from_pretrained("facebook/mms-tts-eng")

# # inference
# # inputs = tokenizer(text_prompt, return_tensors="pt")
# # with torch.no_grad():
# #     output = model(**inputs).waveform

# # # save to wav file
# # scipy.io.wavfile.write("TTS/wav_files/facebook_tts.wav", rate=model.config.sampling_rate, data=output.float().numpy().T)
# # sf.write("wav_files/facebook_tts_2.wav", output.numpy(), samplerate=16000)
# # sf.write("facebook_tts_2.wav", output.float().numpy(), samplerate=16000)

# def generate_file(prompt, file):
#     inputs = tokenizer(prompt, return_tensors="pt")
#     with torch.no_grad():
#         output = model(**inputs).waveform
#     # save to wav file
#     scipy.io.wavfile.write(file, rate=model.config.sampling_rate, data=output.float().numpy().T)

# file_path_long = "TTS/wav_files/auto_facebook_tts_long"
# file_path_short = "TTS/wav_files/auto_facebook_tts_short"

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
def generate_file(prompt, file, tts):
    if "multilingual" in file:
    # if "multilingual" in file or "vctk" in file:
        tts.tts_to_file(
            text=prompt,
            file_path=file,
            language="en",
            speaker_wav="TTS/wav_files/tts_api/tts_models_en_jenny_jenny/long_2.wav"
            )
    else:
        tts.tts_to_file(
            text=prompt,
            file_path=file,
            )

def test_model(model):
    m2 = model.replace("/", "_")
    file_path = "TTS/wav_files/tts_api/"+m2+"/"
    result_path = "TTS/results/"+m2+".txt"

    if os.path.isfile(result_path): # run only models that have no result file
        print("already executed")
        return None

    device = "cuda"
    # model="tts_models/en/jenny/jenny"
    # model="tts_models/en/ljspeech/glow-tts"
    # model="tts_models/en/ek1/tacotron2"
    # model="tts_models/en/ljspeech/speedy-speech"
    # model="tts_models/multilingual/multi-dataset/xtts_v2"
    # model="tts_models/multilingual/multi-dataset/your_tts"
    # model="tts_models/multilingual/multi-dataset/bark"
    # model="tts_models/en/ljspeech/tacotron2-DDC"
    # model="tts_models/en/ljspeech/tacotron2-DDC_ph"
    # model="tts_models/en/ljspeech/tacotron2-DCA"
    # model="tts_models/en/ljspeech/vits"
    # model="tts_models/en/ljspeech/vits--neon"
    # model="tts_models/en/ljspeech/fast_pitch"
    # model="tts_models/en/ljspeech/overflow"
    # model="tts_models/en/ljspeech/neural_hmm"
    # model="tts_models/en/vctk/vits"
    # model="tts_models/en/vctk/fast_pitch"
    # model="tts_models/en/sam/tacotron-DDC"
    # model="tts_models/en/blizzard2013/capacitron-t2-c50"
    # model="tts_models/en/blizzard2013/capacitron-t2-c150_v2"
    # model="tts_models/en/multi-dataset/tortoise-v2"
    # model="tts_models/es/mai/tacotron2-DDC"
    tts = TTS(model, gpu=True).to(device)
    
    try:
        os.makedirs(file_path)
    except:
        # throw execption if path already created, but we don't care about that execption
        # pass
        return None
    file_path_long = file_path + "long_"
    file_path_short = file_path + "short_"

    with open(result_path, "w+") as f:
        f.write("LONG PROMPTS\n")
        exec_times = []
        for i, p in enumerate(text_prompts): 
            start = time.time()
            generate_file(p, file_path_long+str(i)+".wav", tts)
            exec_time = round(time.time()-start, 3)
            exec_times.append(exec_time)
            print("execution_time = ", exec_time)
            f.write(str(exec_time)+"\n")
        f.write("mean exec time = " + str(np.mean(exec_times))+"\n")
        f.write("std exec time = " + str(np.std(exec_times))+"\n")


        f.write("\n\nSHORT PROMPTS\n")
        exec_times = []
        for i, p in enumerate(short_prompts): 
            start = time.time()
            generate_file(p, file_path_short+str(i)+".wav", tts)
            exec_time = round(time.time()-start, 3)
            exec_times.append(exec_time)
            print("short execution_time = ", exec_time)
            f.write(str(exec_time)+"\n")
        f.write("mean exec time = " + str(np.mean(exec_times))+"\n")
        f.write("std exec time = " + str(np.std(exec_times))+"\n")

# list_models = TTS().list_models()._list_models("tts_models")
list_models = ModelManager()._list_models(model_type="tts_models")
black_list = ["bark", "vctk"] # models that generates an error
# take all en or multilingual models that isn't blacklisted
list_en_models = [x for x in list_models if ("/en/" in x or "multilingual" in x)]
list_en_models_not_b = [x for x in list_en_models if all([b not in x for b in black_list])]
print(list_en_models)
print(list_en_models_not_b)
for model in list_en_models_not_b:
    print("execution : ", model)
    test_model(model)