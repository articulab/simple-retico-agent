import retico_core
from retico_core import network, audio, debug, text
from functools import partial
import torch

from dialogue_history import DialogueHistory
from simple_vad import SimpleVADModule
from simple_whisper_asr import SimpleWhisperASRModule
from simple_llm import SimpleLLMModule
from simple_tts import SimpleTTSModule
from simple_speaker import SimpleSpeakerModule


from retico_core.log_utils import (
    filter_has_key,
    filter_does_not_have_key,
    filter_value_in_list,
    filter_value_not_in_list,
    filter_conditions,
    filter_cases,
    configurate_plot,
    plot_once,
)


def main_simple():
    # parameters definition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_folder = "logs/run"
    frame_length = 0.02
    tts_frame_length = 0.2
    rate = 16000
    tts_model_samplerate = 48000
    # model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_S.gguf"
    model_repo = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    model_name = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    system_prompt = "This is a spoken dialog scenario between a teacher and a 8 years old child student.\
        The teacher is teaching mathemathics to the child student.\
        As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        You play the role of a teacher. Here is the beginning of the conversation :"
    plot_config_path = "configs/plot_config_simple.json"
    plot_live = True
    module_order = [
        "Microphone",
        "VAD",
        "ASR",
        "LLM",
        "TTS",
        "Speaker",
    ]
    prompt_format_config = "configs/prompt_format_config.json"
    context_size = 2000

    # filters
    filters = [
        partial(
            filter_cases,
            cases=[
                [("debug", [True])],
                [("level", ["warning", "error"])],
            ],
        )
    ]
    # configurate logger
    terminal_logger, _ = retico_core.log_utils.configurate_logger(
        log_folder, filters=filters
    )

    # configure plot
    configurate_plot(
        is_plot_live=plot_live,
        refreshing_time=1,
        plot_config_path=plot_config_path,
        module_order=module_order,
        window_duration=30,
    )

    dialogue_history = DialogueHistory(
        prompt_format_config,
        terminal_logger=terminal_logger,
        initial_system_prompt=system_prompt,
        context_size=context_size,
    )

    # create modules
    mic = audio.MicrophoneModule()

    vad = SimpleVADModule(
        input_framerate=rate,
        frame_length=frame_length,
    )

    asr = SimpleWhisperASRModule(device=device, framerate=rate)

    llm = SimpleLLMModule(
        model_path=None,
        model_repo=model_repo,
        model_name=model_name,
        dialogue_history=dialogue_history,
        device=device,
        context_size=context_size,
    )

    tts = SimpleTTSModule(
        frame_duration=tts_frame_length,
        device=device,
    )

    speaker = SimpleSpeakerModule(rate=tts_model_samplerate)

    # create network
    mic.subscribe(vad)
    vad.subscribe(asr)
    asr.subscribe(llm)
    llm.subscribe(tts)
    tts.subscribe(speaker)
    speaker.subscribe(vad)

    # running system
    try:
        network.run(mic)
        print("Dialog system running until ENTER key is pressed")
        input()
        network.stop(mic)
    except Exception:
        terminal_logger.exception("exception in main")
        network.stop(mic)
    finally:
        plot_once(
            plot_config_path=plot_config_path,
            module_order=module_order,
        )


if __name__ == "__main__":
    main_simple()
