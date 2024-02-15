# import retico_core
# from retico_core.audio import MicrophoneModule, SpeakerModule
# import wave
# import pyaudio
# import queue
# import time
# from WaveModule import WaveModule

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


#####

# from retico_core import *
# from retico_googleasr import *

msg = []


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
        print(x)
        print(ut)


def callback_google_asr(update_msg):
    # print("lalala")
    # print(update_msg)
    for x, ut in update_msg:
        print(x)
        print(ut)
        print(x.final)


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


# System 2
# from retico_core.text import IncrementalizeASRModule, EndOfUtteranceModule
# from SpeakerModule import SpeakerModule2
from retico_core import *
from retico_core.audio import (
    # AudioDispatcherModule,
    SpeakerModule,
    MicrophoneModule,
)

# from retico_core.debug import CallbackModule
from WaveModule import WaveModule
from retico_googleasr import *


# from retico_whisperasr import WhisperASRModule
import sys

prefix = "/home/mlechape/retico_system_test/"
sys.path.append(prefix + "retico-whisperasr")
from retico_whisperasr.retico_whisperasr.whisperasr import WhisperASRModule

# from retico_whisperasr import WhisperASRModule

# from retico_core.audio import AudioDispatcherModule

wav = WaveModule(file_name="audios/test2.wav")
mic = MicrophoneModule()
# asr = GoogleASRModule("en-US", rate=wav.rate)  # en-US or de-DE or ....
asr = WhisperASRModule()
# end_turn = EndOfUtteranceModule()
# iasr = IncrementalizeASRModule()
cback = debug.CallbackModule(callback=callback)
speaker = SpeakerModule(rate=wav.rate)

wav.subscribe(asr)
wav.subscribe(speaker)
asr.subscribe(cback)

network.run(wav)

print("Running")
input()

network.stop(wav)
