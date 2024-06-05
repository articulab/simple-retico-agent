import os
import keyboard
import torch

from retico_core import *
from retico_core.audio import (
    # AudioDispatcherModule,
    SpeakerModule,
    MicrophoneModule,
)
from utils import *
from microphone_ptt import MicrophonePTTModule
from WozMicrophone_multiple_files import WozMicrophoneModule_multiple_file
from WozMicrophone_one_file import WozMicrophoneModule_one_file
from WozMicrophone_one_file_allinone import WozMicrophoneModule_one_file_allinone
from WaveModule import WaveModule
from WozAsrModule import WozAsrModule
from whisperasr import WhisperASRModule
from whisperasr_2 import WhisperASRModule_2
from llama_cpp_memory_incremental import LlamaCppMemoryIncrementalModule
from coqui_tts import CoquiTTSModule
from SpeakerModule_2 import SpeakerModule_2


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

    # mic = MicrophoneModule(rate=16000, frame_length=0.02)
    # UM ASR LEN =  1764
    # self.framerate =  44100
    # IF =  640

    # mic = WozMicrophoneModule_one_file(frame_length=0.02, log_folder=log_folder)
    mic = WozMicrophoneModule_one_file_allinone(
        frame_length=0.02, log_folder=log_folder
    )
    # UM ASR LEN =  1764
    # self.framerate =  44100
    # IF =  640
    # speaker = SpeakerModule_2(rate=rate, log_folder=log_folder)
    # mic.subscribe(speaker)

    # mic = MicrophoneModule_PTT(rate=16000, frame_length=0.02)

    # print("MIC RATE =", mic.rate)

    # asr = WhisperASRModule(printing=printing, full_sentences=True)
    asr = WhisperASRModule_2(
        printing=printing,
        full_sentences=True,
        input_framerate=16000,
        log_folder=log_folder,
    )
    cback = debug.CallbackModule(callback=callback)
    # llama_mem_icr = LlamaCppMemoryIncrementalModule(
    #     model_path,
    #     None,
    #     None,
    #     None,
    #     system_prompt,
    #     printing=printing,
    #     log_folder=log_folder,
    # )
    # # tts = SpeechBrainTTSModule("en", printing=printing)
    # tts = CoquiTTSModule(
    #     language="en", model="vits_neon", printing=printing, log_folder=log_folder
    # )

    # audio_dispatcher = audio.AudioDispatcherModule(rate=rate)
    # speaker = audio.SpeakerModule(rate=rate)
    # print("TTS SAMPLERATE = ", tts.samplerate)
    # speaker = SpeakerModule_2(rate=tts.samplerate, log_folder=log_folder)

    # # creating network
    # mic.subscribe(speaker)

    # tts.subscribe(mic)
    mic.subscribe(asr)
    asr.subscribe(cback)

    # mic.subscribe(audio_dispatcher)
    # audio_dispatcher.subscribe(asr)

    # asr.subscribe(cback)
    # asr.subscribe(llama_mem_icr)
    # llama_mem_icr.subscribe(tts)
    # tts.subscribe(speaker)

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

    # running system Push To Talk
    # try:
    #     network.run(mic)
    #     print("woz Running")
    #     quit_key = False
    #     while not quit_key:
    #         if keyboard.is_pressed("q"):
    #             quit_key = True
    #         time.sleep(1)
    #     # input()
    #     network.stop(mic)
    #     merge_logs(log_folder)
    # except Exception as err:
    #     print(f"Unexpected {err=}, {type(err)=}")
    #     network.stop(mic)


# def main_demo():
#     """
#     The `main_woz` function sets up a spoken dialog scenario between a teacher and an 8-year-old student
#     for teaching mathematics using various modules for audio input, speech recognition, memory
#     processing, text-to-speech, and audio output.

#     It uses a WozMicrophoneModule which plays previously recorded wav files as if it was audio captured by a microphone in real time.
#     It is used to test the latency of the system with fixed audio inputs.
#     """

#     # LLM info
#     model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_S.gguf"

#     system_prompt = b"This is a spoken dialog scenario between a teacher and a 8 years old child student.\
#         The teacher is teaching mathemathics to the child student.\
#         As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
#         You play the role of a teacher. Here is the beginning of the conversation :"

#     printing = False
#     rate = 16000

#     # create log folder
#     log_folder = create_new_log_folder("logs/test/16k/Recording (1)/demo")

#     # Instantiate Modules
#     mic = MicrophoneModule_PTT(rate=rate, frame_length=0.02)
#     # asr = WhisperASRModule(printing=printing, full_sentences=True)
#     asr = WhisperASRModule_2(
#         printing=printing,
#         full_sentences=True,
#         input_framerate=rate,
#         log_folder=log_folder,
#     )
#     cback = debug.CallbackModule(callback=callback)
#     llama_mem_icr = LlamaCppMemoryIncrementalModule(
#         model_path,
#         None,
#         None,
#         None,
#         system_prompt,
#         printing=printing,
#         log_folder=log_folder,
#     )
#     tts = CoquiTTSModule(
#         language="en", model="vits_neon", printing=printing, log_folder=log_folder
#     )
#     speaker = SpeakerModule_2(rate=tts.samplerate, log_folder=log_folder)

#     # Create network
#     mic.subscribe(asr)
#     # asr.subscribe(cback)
#     asr.subscribe(llama_mem_icr)
#     llama_mem_icr.subscribe(tts)
#     tts.subscribe(speaker)

#     # running system Push To Talk
#     try:
#         network.run(mic)
#         print("Running")
#         quit_key = False
#         while not quit_key:
#             if keyboard.is_pressed("q"):
#                 quit_key = True
#             time.sleep(1)
#         network.stop(mic)
#         merge_logs(log_folder)
#     except Exception as err:
#         print(f"Unexpected {err=}, {type(err)=}")
#         network.stop(mic)


def main_demo():
    """
    The `main_demo` function creates and runs a dialog system that is able to have a conversation with the user.

    The dialog system is composed of different modules:
    - a Microphone : captures the user's voice
    - an ASR : transcribes the user's voice into text
    - a LLM : generates a textual answer to the trancription from user's spoken sentence.
    - a TTS : generates a spoken answer from the LLM's textual answer.
    - a Speaker : outputs the spoken answer generated by the system.

    We provide the system with a scenario (contained in the "system_prompt") that it will follow through the conversation :
    The system is a teacher and it will teach mathematics to a 8-year-old child student (the user)

    the parameters defined :
    - model_path : the path to the weights of the LLM that will be used in the dialog system.
    - system_prompt : a part of the prompt that will be given to the LLM at every agent turn to set the scenario of the conversation.
    - printing : an argument that set to True will print a lot of information useful for degugging.
    - rate : the target audio signal rate to which the audio captured by the microphone will be converted to (so that it is suitable for every module)
    - frame_length : the chosen frame length in seconds at which the audio signal will be chunked.
    - log_folder : the path to the folder where the logs (information about each module's latency) will be saved.

    It is recommended to not modify the rate and frame_length parameters because the modules were coded with theses values
    and it is not ensured that the system will run correctly with other values.
    """

    # parameters definition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cuda"
    # device = "cpu"
    printing = False
    log_folder = create_new_log_folder("logs/run")
    frame_length = 0.02
    rate = 16000
    tts_model_samplerate = 22050
    tts_model = "vits_neon"
    model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_S.gguf"
    system_prompt = b"This is a spoken dialog scenario between a teacher and a 8 years old child student.\
        The teacher is teaching mathemathics to the child student.\
        As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        You play the role of a teacher. Here is the beginning of the conversation :"

    # create modules
    mic = MicrophonePTTModule(rate=rate, frame_length=frame_length)
    asr = WhisperASRModule_2(
        device=device,
        printing=printing,
        full_sentences=True,
        input_framerate=rate,
        log_folder=log_folder,
    )
    llama_mem_icr = LlamaCppMemoryIncrementalModule(
        model_path,
        None,
        None,
        None,
        system_prompt,
        printing=printing,
        log_folder=log_folder,
        device=device,
    )
    tts = CoquiTTSModule(
        language="en",
        model=tts_model,
        printing=printing,
        log_folder=log_folder,
        device=device,
    )

    speaker = SpeakerModule_2(rate=tts_model_samplerate, log_folder=log_folder)

    # create network
    mic.subscribe(asr)
    asr.subscribe(llama_mem_icr)
    llama_mem_icr.subscribe(tts)
    tts.subscribe(speaker)

    # running network with the Push To Talk system
    try:
        network.run(mic)
        print("Dialog system ready")
        keyboard.wait("q")
        network.stop(mic)
        merge_logs(log_folder)
    except Exception as err:
        print(f"Unexpected {err}")
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
    main_demo()
    # merge_logs("logs/test/16k/Recording (1)/demo_4")
