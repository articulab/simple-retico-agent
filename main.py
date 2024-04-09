from asyncio import Queue
import csv
import datetime
import os
import sys

from SpeakerModule_2 import SpeakerModule_2
from WozMicrophone_multiple_files import WozMicrophoneModule_multiple_file
from WozMicrophone_one_file import WozMicrophoneModule_one_file
from utils import *
from whisperasr_2 import WhisperASRModule_2

prefix = "/home/mlechape/retico_system_test/"
sys.path.append(prefix + "retico-whisperasr")
from CoquiTTSModule import CoquiTTSModule
from WaveModule import WaveModule
from whisperasr import WhisperASRModule
from WozAsrModule import WozAsrModule

# from speechbraintts import SpeechBrainTTSModule
from llama import LlamaModule
from llama_cpp_chat import LlamaCppModule
from llama_cpp_memory import LlamaCppMemoryModule
from llama_cpp_memory_incremental import LlamaCppMemoryIncrementalModule
from llama_chat import LlamaChatModule

from retico_core import *
from retico_core.audio import (
    # AudioDispatcherModule,
    SpeakerModule,
    MicrophoneModule,
)


def callback(update_msg):
    global msg
    for x, ut in update_msg:
        if ut == UpdateType.ADD:
            msg.append(x)
        if ut == UpdateType.REVOKE:
            msg.remove(x)
    txt = ""
    committed = False
    for x in msg:
        # if x.final:
        #     print("final commited" + str(x.committed))
        txt += x.text + " "
        committed = committed or x.committed
    print(" " * 80, end="\r")
    print(f"{txt}", end="\r")
    if committed:
        msg = []
        print("")


def callback2(update_msg):
    # print("lalala")
    # print(update_msg)
    for x, ut in update_msg:
        # print(x)
        # print(ut)
        print(str(x.payload) + " " + str(ut))


def callback_ponct(update_msg):
    global SENTENCE
    ponctuation = [".", ",", "?", "!", ":", ";"]
    for x, ut in update_msg:
        if ut == UpdateType.COMMIT:
            SENTENCE += str(x.payload) + " "
            # print(SENTENCE)
            if str(x.payload)[-1] in ponctuation:
                print(SENTENCE)
                SENTENCE = ""


def callback_stream_ponct(update_msg):
    global SENTENCE
    is_ponct = False
    ponctuation = [".", ",", "?", "!", ":", ";"]
    for x, ut in update_msg:
        # if ut == UpdateType.ADD:
        #     SENTENCE += str(x.payload) + " "
        if ut == UpdateType.COMMIT:
            SENTENCE += str(x.payload) + " "
            is_ponct = str(x.payload)[-1] in ponctuation

    # Clear the console to reprint the updated transcription.
    os.system("cls" if os.name == "nt" else "clear")
    print(SENTENCE)
    # Flush stdout.
    print("", end="", flush=True)
    if is_ponct:
        SENTENCE = ""


def callback_google_asr(update_msg):
    # print("lalala")
    # print(update_msg)
    for x, ut in update_msg:
        print(x)
        print(ut)
        print(x.final)


# microphone_module = MicrophoneModule()
# speaker_module = SpeakerModule()

# microphone_module.subscribe(speaker_module)

# This Plays the audio input by the mic into the speakers

# retico_core.network.run(microphone_module)
# # Wait for an input from the user
# input()
# retico_core.network.stop(microphone_module)


# I am trying to create a way to use a wav file instead of the microphone as the audio input.


# This Plays the audio input by the mic into the speakers
# r_files = wave.open("test.wav", "rb")
# a = r_files.readframes(10)
# p = pyaudio.PyAudio()
# chunk = 1024
# stream = p.open(format = p.get_format_from_width(r_files.getsampwidth()),
#                 channels = r_files.getnchannels(),
#                 rate = r_files.getframerate(),
#                 output = True)
# data = r_files.readframes(chunk)
# while data != '':
#     stream.write(data)
#     data = r_files.readframes(chunk)


# r_files = wave.open("test.wav", "rb")
# chunk = 1024
# audio_buffer = queue.Queue()
# sleeping_time = 1
# data = r_files.readframes(chunk)
# while data != '':
#     audio_buffer.put(data)
#     time.sleep(sleeping_time)
#     data = r_files.readframes(chunk)


# wave_module = WaveModule("test.wav")
# speaker_module = SpeakerModule(rate=wave_module.rate)
# wave_module.subscribe(speaker_module)

# microphone_module = MicrophoneModule()
# speaker_module = SpeakerModule()
# microphone_module.subscribe(speaker_module)

# retico_core.network.save(wave_module, "wav_to_speaker_net")

# retico_core.network.load_and_execute("wav_to_speaker_net.rtc")


# System 1

# m1 = WaveModule(file_name="audios/test2.wav")
# m2 = GoogleASRModule("en-US", rate=m1.rate)  # en-US or de-DE or ....
# m3 = debug.CallbackModule(callback=callback)
# m4 = SpeakerModule(rate=m1.rate)

# # m1.subscribe(m3)
# m1.subscribe(m2)
# m2.subscribe(m3)
# m1.subscribe(m4)


# network.run(m1)

# print("Running")
# input()

# network.stop(m1)

# export GOOGLE_APPLICATION_CREDENTIALS=/home/mlechape/retico_system_test/Google_API_Key.json

# retico_core.network.save(m1, "networks/wav_asr_debug_speaker_net")

# retico_core.network.load_and_execute("networks/wav_asr_debug_speaker_net.rtc")

# # from retico_whisperasr import WhisperASRModule

# # from retico_core.audio import AudioDispatcherModule

# # wav = WaveModule(file_name="audios/test2.wav")
# mic = MicrophoneModule()
# # asr = GoogleASRModule("en-US", rate=wav.rate)  # en-US or de-DE or ....
# asr = WhisperASRModule()
# # end_turn = EndOfUtteranceModule()
# # iasr = IncrementalizeASRModule()
# # cback = debug.CallbackModule(callback=callback)
# tts = SpeechBrainTTSModule("en")
# speaker = audio.SpeakerModule(rate=22050)
# # speaker = SpeakerModule(rate=mic.rate)
# # woz = WozAsrModule()

# # model_name = "meta-llama/Llama-2-7b-hf"
# model_name = 'mediocredev/open-llama-3b-v2-chat'
# # chat_history = [
# #     {"role": "user", "content": "Hello"},
# #     {"role": "assistant", "content": "Hello! I am your math teacher and I will teach you addition today."},
# #     {"role": "user", "content": "I am your 8 years old child student and I can't wait to learn about mathematics !"},
# # ]
# chat_history = [
#     {"role": "user", "content": "Hello"},
#     {"role": "assistant", "content": "Hello! I am your math teacher, you are a 8 years old student. This is a dialog during which I will teach you how to add two numbers together."},
# ]
# # llama = LlamaModule(model_name)
# llama_chat = LlamaChatModule(model_name, chat_history=chat_history)

# # wav.subscribe(asr)
# # wav.subscribe(speaker)
# # # asr.subscribe(cback)
# # woz.subscribe(llama_chat)
# mic.subscribe(asr)
# # asr.subscribe(cback)
# asr.subscribe(llama_chat)
# llama_chat.subscribe(tts)
# tts.subscribe(speaker)

# network.run(mic)

# print("Running")
# input()

# network.stop(mic)


SENTENCE = ""


def main_llama_cpp_python_chat_7b():

    # Usable model
    # model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_S.gguf"
    # model_path = "./models/mistral-7b-v0.1.Q4_K_S.gguf"
    # model_path = "./models/zephyr-7b-beta.Q4_0.gguf"
    # model_path = "./models/llama-2-13b-chat.Q4_K_S.gguf"
    # model_path = "./models/llama-2-13b.Q4_K_S.gguf"

    # LLM info
    model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_S.gguf"
    chat_history = [
        {
            "role": "system",
            "content": "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
            The teacher is teaching mathemathics to the child student. \
            As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
            You play the role of a teacher. Here is the beginning of the conversation :",
        },
        {"role": "user", "content": "Hello !"},
        {"role": "assistant", "content": "Hi! How are your today ?"},
        {
            "role": "user",
            "content": "I am fine, and I can't wait to learn mathematics !",
        },
    ]

    initial_prompt = b"<<SYS>>This is a spoken dialog scenario between a teacher and a 8 years old child student.\
        The teacher is teaching mathemathics to the child student.\
        As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
            You play the role of a teacher. Here is the beginning of the conversation :<</SYS>>\
        [INST]Child : Hello ![/INST]\
        Teacher : Hi! How are your today ?\
        [INST]Child : I am fine, and I can't wait to learn mathematics ![/INST]\
        Teacher : Great news! I'm excited to help you learn as well. Let's start with something simple. Can you tell me what comes before the number five ?\
        [INST]Child : four ?[/INST]\
        Teacher : Excellent job! Keep it up. Let's move on to something a little more challenging. Can you tell me how many apples you have if I give you two baskets, each having three apples?\
        [INST]Child : six ?[/INST]\
        Teacher : Correct! Now let's see if you can find 3 + 3. Great work! Keep going like this and we will learn maths together step by step.\
        [INST]Child : okay.[/INST]\
        Teacher: That's the spirit! Let's continue practicing these simple addition problems together. If you ever feel confused or have any doubts, please don't hesitate to ask questions.\
        [INST]Child : yeah ! let's do another exercice.[/INST]\
        Teacher: "

    system_prompt = b"This is a spoken dialog scenario between a teacher and a 8 years old child student.\
        The teacher is teaching mathemathics to the child student.\
        As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        You play the role of a teacher. Here is the beginning of the conversation :"

    printing = True

    # creating modules
    mic = MicrophoneModule()
    asr = WhisperASRModule(printing=printing, full_sentences=True)
    # llama_cpp = LlamaCppModule(model_path, chat_history=chat_history)
    # llama_mem = LlamaCppMemoryModule(model_path, None, None, initial_prompt)
    llama_mem_icr = LlamaCppMemoryIncrementalModule(
        model_path, None, None, None, system_prompt, printing=printing
    )
    # tts = SpeechBrainTTSModule("en", printing=printing)
    tts = CoquiTTSModule(language="en", model="vits_neon", printing=printing)
    speaker = audio.SpeakerModule(
        rate=tts.samplerate
    )  # Why does the speaker module have to copy the tts rate ?
    cback = debug.CallbackModule(callback=callback2)

    # creating network
    mic.subscribe(asr)
    asr.subscribe(cback)
    asr.subscribe(llama_mem_icr)
    llama_mem_icr.subscribe(tts)
    tts.subscribe(speaker)

    # running system
    network.run(mic)
    print("Running")
    input()
    network.stop(mic)


def main_woz():
    """
    The `main_woz` function sets up a spoken dialog scenario between a teacher and an 8-year-old student
    for teaching mathematics using various modules for audio input, speech recognition, memory
    processing, text-to-speech, and audio output.

    It uses a WozMicrophoneModule which plays previously recorded wav files as if it was audio captured by a microphone in real time.
    It is used to test the latency of the system with fixed audio inputs.
    """

    # LLM info
    model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_S.gguf"

    system_prompt = b"This is a spoken dialog scenario between a teacher and a 8 years old child student.\
        The teacher is teaching mathemathics to the child student.\
        As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        You play the role of a teacher. Here is the beginning of the conversation :"

    printing = True
    # rate = 96000
    rate = 16000
    # rate = 32000

    # create log folder
    log_folder = create_new_log_folder("logs/test/16k/Recording (1)/demo")

    # assert chunk_size = frame_rate * frame_length * nb_channels  <= 960

    # creating modules
    # mic = WozMicrophoneModule(folder_path="audios/8k/", rate=rate * 2)
    # mic = WozMicrophoneModule_2(folder_path="audios/stereo/48k/", rate=rate * 2)
    # mic = WozMicrophoneModule_one_file(
    #     file="audios/stereo/48k/Recording (5).wav", frame_length=0.02
    # )

    # mic = WozMicrophoneModule_multiple_file(
    #     folder_path="audios/stereo/48k/", frame_length=0.015
    # )
    # UM ASR LEN =  5760
    # self.framerate =  96000
    # IF =  960

    # mic = WozMicrophoneModule_multiple_file(
    #     folder_path="audios/stereo/8k/", frame_length=0.015
    # )
    # UM ASR LEN =  960
    # self.framerate =  16000

    # mic = WozMicrophoneModule_multiple_file(
    #     folder_path="audios/stereo/16k/", frame_length=0.015
    # )
    # UM ASR LEN =  1920
    # self.framerate =  32000
    # IF =  960

    # mic = WozMicrophoneModule_multiple_file(
    #     folder_path="audios/mono/44k/", frame_length=0.02
    # )
    # UM ASR LEN =  1764
    # self.framerate =  44100
    # IF =  640

    mic = MicrophoneModule(rate=16000, frame_length=0.02)
    # UM ASR LEN =  1764
    # self.framerate =  44100
    # IF =  640

    # mic = WozMicrophoneModule_one_file(frame_length=0.02, log_folder=log_folder)
    # UM ASR LEN =  1764
    # self.framerate =  44100
    # IF =  640

    # print("MIC RATE =", mic.rate)

    # asr = WhisperASRModule(printing=printing, full_sentences=True)
    asr = WhisperASRModule_2(
        printing=printing,
        full_sentences=True,
        input_framerate=16000,
        log_folder=log_folder,
    )
    cback = debug.CallbackModule(callback=callback)
    llama_mem_icr = LlamaCppMemoryIncrementalModule(
        model_path,
        None,
        None,
        None,
        system_prompt,
        printing=printing,
        log_folder=log_folder,
    )
    # tts = SpeechBrainTTSModule("en", printing=printing)
    tts = CoquiTTSModule(
        language="en", model="vits_neon", printing=printing, log_folder=log_folder
    )

    # audio_dispatcher = audio.AudioDispatcherModule(rate=rate)
    # speaker = audio.SpeakerModule(rate=rate)
    print("TTS SAMPLERATE = ", tts.samplerate)
    speaker = SpeakerModule_2(rate=tts.samplerate, log_folder=log_folder)

    # creating network
    # mic.subscribe(speaker)

    # tts.subscribe(mic)
    mic.subscribe(asr)

    # mic.subscribe(audio_dispatcher)
    # audio_dispatcher.subscribe(asr)

    # asr.subscribe(cback)
    asr.subscribe(llama_mem_icr)
    llama_mem_icr.subscribe(tts)
    tts.subscribe(speaker)

    # running system
    try:
        network.run(mic)
        print("woz Running")
        input()
        network.stop(mic)
        merge_logs(log_folder)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        network.stop(mic)


def main_llama_chat_3b():

    # LLM info
    model_name = "mediocredev/open-llama-3b-v2-chat"
    chat_history = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "Hello! I am your math teacher, you are a 8 years old student. This is a dialog during which I will teach you how to add two numbers together.",
        },
    ]

    # creating modules
    mic = MicrophoneModule()
    asr = WhisperASRModule()
    llama_chat = LlamaChatModule(model_name, chat_history=chat_history)
    tts = SpeechBrainTTSModule("en")
    speaker = audio.SpeakerModule(
        rate=tts.samplerate
    )  # Why does the speaker module have to copy the tts rate ?

    # creating network
    mic.subscribe(asr)
    asr.subscribe(llama_chat)
    llama_chat.subscribe(tts)
    tts.subscribe(speaker)

    # running system
    try:
        network.run(mic)
        print("Running")
        input()
        network.stop(mic)
    except:
        network.stop(mic)


msg = []

if __name__ == "__main__":
    # main_llama_chat_3b()
    # main_llama_cpp_python_chat_7b()
    # main_woz()
    merge_logs("logs/test/16k/Recording (1)/demo_1")
