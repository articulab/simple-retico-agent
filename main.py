import keyboard
import retico_core.abstract
import torch

from retico_core import *

from amq import (
    AMQReader,
    AMQWriter,
    AMQWriterOpening,
    TextAnswertoBEATBridge,
    fakeBEATSARA,
    fakeTTSSARA,
)
from vad_turn import VADTurnModule
from whisper_asr_interruption import WhisperASRInterruptionModule
from llama_cpp_memory_incremental_interruption import (
    LlamaCppMemoryIncrementalInterruptionModule,
)
from coqui_tts_interruption import CoquiTTSInterruptionModule
from speaker_interruption import SpeakerInterruptionModule

from whisper_asr import WhisperASRModule
from llama_cpp_memory_incremental import LlamaCppMemoryIncrementalModule
from coqui_tts import CoquiTTSModule
from speaker_2 import SpeakerModule_2
from microphone_ptt import MicrophonePTTModule

from utils import *
from woz_audio.WozMicrophone_multiple_files import WozMicrophoneModule_multiple_file
from woz_audio.WozMicrophone_one_file import WozMicrophoneModule_one_file
from woz_audio.WozMicrophone_one_file_allinone import (
    WozMicrophoneModule_one_file_allinone,
)
from woz_audio.WaveModule import WaveModule
from woz_audio.WozAsrModule import WozAsrModule

from retico_zmq.retico_zmq.zmq import WriterSingleton, ZeroMQWriter, ReaderSingleton


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
    asr = WhisperASRModule(
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
    asr = WhisperASRModule(
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


def main_speaker_interruption():
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
    tts_frame_length = 0.2
    rate = 16000
    tts_model_samplerate = 22050
    # tts_model = "vits_neon"
    tts_model = "vits_vctk"
    model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_S.gguf"
    system_prompt = b"This is a spoken dialog scenario between a teacher and a 8 years old child student.\
        The teacher is teaching mathemathics to the child student.\
        As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        You play the role of a teacher. Here is the beginning of the conversation :"

    # create modules
    # mic = MicrophonePTTModule(rate=rate, frame_length=frame_length)
    mic = audio.MicrophoneModule(rate=rate, frame_length=frame_length)

    asr = WhisperASRInterruptionModule(
        device=device,
        printing=printing,
        full_sentences=True,
        input_framerate=rate,
        log_folder=log_folder,
    )

    llama_mem_icr = LlamaCppMemoryIncrementalInterruptionModule(
        model_path,
        None,
        None,
        None,
        system_prompt,
        printing=printing,
        log_folder=log_folder,
        device=device,
    )

    tts = CoquiTTSInterruptionModule(
        language="en",
        model=tts_model,
        printing=printing,
        log_folder=log_folder,
        frame_duration=tts_frame_length,
        device=device,
    )

    speaker = SpeakerInterruptionModule(
        rate=tts_model_samplerate, log_folder=log_folder
    )

    vad = VADTurnModule(
        printing=printing,
        input_framerate=rate,
        log_folder=log_folder,
        frame_length=frame_length,
    )

    # create network
    mic.subscribe(vad)
    asr.subscribe(llama_mem_icr)
    llama_mem_icr.subscribe(tts)
    tts.subscribe(speaker)

    vad.subscribe(asr)
    vad.subscribe(llama_mem_icr)
    vad.subscribe(tts)
    vad.subscribe(speaker)
    speaker.subscribe(llama_mem_icr)
    speaker.subscribe(vad)

    # running system
    try:
        network.run(mic)
        print("Dialog system ready")
        keyboard.wait("q")
        network.stop(mic)
        merge_logs(log_folder)
    except (
        Exception,
        NotImplementedError,
        ValueError,
        AttributeError,
        AssertionError,
    ) as err:
        print(f"Unexpected {err}")
        network.stop(mic)


def test_cuda():
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
    llama_mem_icr = LlamaCppMemoryIncrementalInterruptionModule(
        model_path,
        None,
        None,
        None,
        system_prompt,
        printing=printing,
        log_folder=log_folder,
        device=device,
    )
    # running network with the Push To Talk system
    try:
        network.run(llama_mem_icr)
        print("Dialog system ready")
        keyboard.wait("q")
        network.stop(llama_mem_icr)
        merge_logs(log_folder)
    except Exception as err:
        print(f"Unexpected {err}")
        network.stop(llama_mem_icr)


def callback_fun(update_msg):
    for x in update_msg:
        print("callback = ", x)


def callback_only_beat(update_msg):
    for x in update_msg:
        print("callback = ", x)


def test_body():
    """
    Testing if an ActiveMQ retico module (AMQWriterOpening) could trigger SARA's oppening animation.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cuda"
    # device = "cpu"
    printing = False
    log_folder = create_new_log_folder("logs/run")
    frame_length = 0.02
    rate = 16000

    hosts = [("localhost", 61613)]
    # ip = "127.0.0.1"
    ip = "localhost"
    port = "61613"

    mic = audio.MicrophoneModule(rate=rate, frame_length=frame_length)

    asr = WhisperASRInterruptionModule(
        device=device,
        printing=printing,
        full_sentences=True,
        input_framerate=rate,
        log_folder=log_folder,
    )

    vad = VADTurnModule(
        printing=printing,
        input_framerate=rate,
        log_folder=log_folder,
        frame_length=frame_length,
    )

    amq = AMQWriterOpening(ip=ip, port=port, topic="test")

    # # the IP of the writer should be the originating PC's IP
    # WriterSingleton(ip=ip, port=port)
    # test = ZeroMQWriter(topic="test")

    # # the IP of the reader should be the target PC's IP
    # reader = ReaderSingleton(ip=ip, port=port)
    # reader.add(topic="test", target_iu_type=retico_core.text.TextIU)

    cback = debug.CallbackModule(callback=callback_fun)

    mic.subscribe(vad)
    vad.subscribe(asr)
    asr.subscribe(amq)
    # asr.subscribe(test)
    # asr.subscribe(reader)
    # reader.subscribe(cback)

    # running system
    try:
        network.run(mic)
        print("Dialog system ready")
        keyboard.wait("q")
        network.stop(mic)
        # merge_logs(log_folder)
    except (
        Exception,
        NotImplementedError,
        ValueError,
        AttributeError,
        AssertionError,
    ) as err:
        print(f"Unexpected {err}")
        network.stop(mic)


def test_body_2():
    """
    ADDITION : testing if sound and further dialogue animations could be triggered by ActiveMQ retico modules (using fake BEAT and TTS modules).
    Testing SARA's body activation with a retico module simulating BEAT output (AMQWriter) and another simulating TTS output (fakeTTSSARA).
    The fakeTTS module plays the speech.mp3 file, that is not the file matching the gesture.

    SARA's body is moving every time the ASR is recognizing voice, and sending a message to the fake BEAT module.
    -> Which means that it is possible to control SARA's body entirely with a real BEAT-like module implemented in retico
    (and with retico modules sending data to the unity agent through an MQ like ActiveMQ).
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cuda"
    # device = "cpu"
    printing = False
    log_folder = create_new_log_folder("logs/run")
    frame_length = 0.02
    rate = 16000

    ip = "localhost"
    port = "61613"

    mic = audio.MicrophoneModule(rate=rate, frame_length=frame_length)

    vad = VADTurnModule(
        printing=printing,
        input_framerate=rate,
        log_folder=log_folder,
        frame_length=frame_length,
    )

    asr = WhisperASRInterruptionModule(
        device=device,
        printing=printing,
        full_sentences=True,
        input_framerate=rate,
        log_folder=log_folder,
    )

    beat = fakeBEATSARA(ip=ip, port=port)

    mark_path = "bodies/sara/mark.json"
    speech_path = "bodies/sara/speech.mp3"
    tts = fakeTTSSARA(
        speech_file_path=speech_path, mark_file_path=mark_path, ip=ip, port=port
    )

    cback = debug.CallbackModule(callback=callback_fun)

    mic.subscribe(vad)
    vad.subscribe(asr)
    asr.subscribe(beat)
    tts.subscribe(cback)
    # asr.subscribe(opening)
    # amqr.subscribe(tts)
    # amqr.subscribe(cback)

    # running system
    try:
        network.run(mic)
        print("Dialog system ready")
        keyboard.wait("q")
        network.stop(mic)
        # merge_logs(log_folder)
    except (
        Exception,
        NotImplementedError,
        ValueError,
        AttributeError,
        AssertionError,
    ) as err:
        print(f"Unexpected {err}")
        network.stop(mic)


def test_body_3():
    """
    ADDITION : using real BEAT module instead of a fake BEAT outputting the same message each trigger.
    Testing SARA's body activation using a retico pipeline and BEAT for the gesture generation.
    The retico pipeline is MIC - VAD - ASR, with 3 additional modules :
    - TextAnswertoBEATBridge : a module creating a bridge between the ASR and BEAT
    - AMQWriter : a module that sends the IUs to ActiveMQ (to communicate with BEAT and SARA body).
    module simulating BEAT output (AMQWriter)
    - fakeTTSSARA : a module that simulates SARA's TTS (fakeTTSSARA). The fakeTTS module plays the speech.mp3 file, that is not the file matching the gesture, but it is useful because the gestures are not played without receiving activeMQ message that says that the agent is talking.

    SARA's body is moving with the message received by BEAT module.
    -> Which means that it is possible to control SARA's body entirely with a real BEAT-like module implemented in retico
    (and with retico modules sending data to the unity agent through an MQ like ActiveMQ).
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    printing = False
    log_folder = create_new_log_folder("logs/run")
    frame_length = 0.02
    rate = 16000

    ip = "localhost"
    port = "61613"

    mic = audio.MicrophoneModule(rate=rate, frame_length=frame_length)

    vad = VADTurnModule(
        printing=printing,
        input_framerate=rate,
        log_folder=log_folder,
        frame_length=frame_length,
    )

    asr = WhisperASRInterruptionModule(
        device=device,
        printing=printing,
        full_sentences=True,
        input_framerate=rate,
        log_folder=log_folder,
    )

    txt_to_beat = TextAnswertoBEATBridge()

    aw = AMQWriter(ip=ip, port=port)

    # beat = fakeBEATSARA(ip=ip, port=port)

    mark_path = "bodies/sara/mark.json"
    speech_path = "bodies/sara/speech.mp3"
    fake_tts = fakeTTSSARA(
        speech_file_path=speech_path, mark_file_path=mark_path, ip=ip, port=port
    )

    cback = debug.CallbackModule(callback=callback_fun)

    # 1rst network (without NLG, or real TTS) the user utterance is sent to BEAT to generate gestures, and a fake TTS module speaks a fixed mp3 file to run the gestures.
    mic.subscribe(vad)
    vad.subscribe(asr)
    asr.subscribe(txt_to_beat)
    txt_to_beat.subscribe(aw)
    fake_tts.subscribe(cback)
    # asr.subscribe(opening)
    # amqr.subscribe(tts)
    # amqr.subscribe(cback)

    # running system
    try:
        network.run(mic)
        print("Dialog system ready")
        keyboard.wait("q")
        network.stop(mic)
        # merge_logs(log_folder)
    except (
        Exception,
        NotImplementedError,
        ValueError,
        AttributeError,
        AssertionError,
    ) as err:
        print(f"Unexpected {err}")
        network.stop(mic)


def test_body_4():
    """
    Testing SARA's body activation using a retico pipeline and BEAT for the gesture generation.
    The retico pipeline is MIC - VAD - ASR, with 3 additional modules :
    - TextAnswertoBEATBridge : a module creating a bridge between the ASR and BEAT
    - AMQWriter : a module that sends the IUs to ActiveMQ (to communicate with BEAT and SARA body).
    module simulating BEAT output (AMQWriter)
    - fakeTTSSARA : a module that simulates SARA's TTS (fakeTTSSARA). The fakeTTS module plays the speech.mp3 file, that is not the file matching the gesture, but it is useful because the gestures are not played without receiving activeMQ message that says that the agent is talking.

    SARA's body is moving with the message received by BEAT module.
    -> Which means that it is possible to control SARA's body entirely with a real BEAT-like module implemented in retico
    (and with retico modules sending data to the unity agent through an MQ like ActiveMQ).

    TODO:
    - integrate a NLG module to create an textual answer
    - integrate a real TTS retico module in the pipeline, to synthetise voice from the textual answer
    - integrate a speaker module in the pipeline
    - make the speaker module send ActiveMQ messages to SARA's body when it plays the sound to verify that it can be synchronized.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cuda"
    # device = "cpu"
    printing = False
    log_folder = create_new_log_folder("logs/run")
    frame_length = 0.02
    tts_frame_length = 0.2
    rate = 16000
    tts_model_samplerate = 22050
    tts_model = "vits_vctk"
    model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_S.gguf"
    system_prompt = b"This is a spoken dialog scenario between a teacher and a 8 years old child student.\
        The teacher is teaching mathemathics to the child student.\
        As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        You play the role of a teacher. Here is the beginning of the conversation :"

    ip = "localhost"
    port = "61613"

    # mic = audio.MicrophoneModule(rate=rate, frame_length=frame_length)
    mic = MicrophonePTTModule(rate=rate, frame_length=frame_length)

    vad = VADTurnModule(
        printing=printing,
        input_framerate=rate,
        log_folder=log_folder,
        frame_length=frame_length,
    )

    asr = WhisperASRInterruptionModule(
        device=device,
        printing=printing,
        full_sentences=True,
        input_framerate=rate,
        log_folder=log_folder,
    )

    llama_mem_icr = LlamaCppMemoryIncrementalInterruptionModule(
        model_path,
        None,
        None,
        None,
        system_prompt,
        printing=printing,
        log_folder=log_folder,
        device=device,
    )

    txt_to_beat = TextAnswertoBEATBridge()

    aw = AMQWriter(ip=ip, port=port)

    # ar = AMQReader(ip=ip, port=port)
    # ar.add(destination="/topic/DEFAULT_SCOPE", target_iu_type=retico_core.text.TextIU)

    # tts = CoquiTTSInterruptionModule(
    #     language="en",
    #     model=tts_model,
    #     printing=printing,
    #     log_folder=log_folder,
    #     frame_duration=tts_frame_length,
    #     device=device,
    # )

    # speaker = SpeakerInterruptionModule(
    #     rate=tts_model_samplerate, log_folder=log_folder
    # )

    # cback = debug.CallbackModule(callback=callback_fun)

    # network
    mic.subscribe(vad)
    vad.subscribe(asr)
    asr.subscribe(llama_mem_icr)
    llama_mem_icr.subscribe(txt_to_beat)
    txt_to_beat.subscribe(aw)
    # ar.subscribe(tts)
    # tts.subscribe(speaker)

    # running system
    try:
        network.run(mic)
        print("Dialog system ready")
        keyboard.wait("q")
        network.stop(mic)
        # merge_logs(log_folder)
    except (
        Exception,
        NotImplementedError,
        ValueError,
        AttributeError,
        AssertionError,
    ) as err:
        print(f"Unexpected {err}")
        network.stop(mic)


import importlib


def test_structlog():
    audio = importlib.import_module("retico-core.retico_core.audio")
    absctract = importlib.import_module("retico-core.retico_core.absctract")
    mic = audio.MicrophoneModule()
    speakers = audio.SpeakerModule()

    mic.subscribe(speakers)
    # running system
    try:
        absctract.network.run(mic)
        print("Dialog system ready")
        keyboard.wait("q")
        absctract.network.stop(mic)
        # merge_logs(log_folder)
    except (
        Exception,
        NotImplementedError,
        ValueError,
        AttributeError,
        AssertionError,
    ) as err:
        print(f"Unexpected {err}")
        absctract.network.stop(mic)


msg = []

if __name__ == "__main__":
    # main_llama_cpp_python_chat_7b()
    # main_woz()
    # main_demo()
    # main_speaker_interruption()
    # test_cuda()
    # merge_logs("logs/test/16k/Recording (1)/demo_4")

    # test_body()
    # test_body_2()
    # test_body_3()
    # test_body_4()

    test_structlog()
