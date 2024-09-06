from keyboard import wait
from retico_core import *
from retico_core.abstract import *
from retico_core.text import *
import structlog
import torch

from amq import TextAnswertoBEATBridge
from vad_turn import VADTurnModule
from whisper_asr_interruption import WhisperASRInterruptionModule


def test_structlog():
    logger = structlog.get_logger()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    printing = False
    frame_length = 0.02
    rate = 16000

    mic = audio.MicrophoneModule()
    vad = VADTurnModule(
        printing=printing,
        input_framerate=rate,
        log_folder="logs/",
        frame_length=frame_length,
    )
    asr = WhisperASRInterruptionModule(
        device=device,
        printing=False,
        full_sentences=True,
        input_framerate=16000,
        log_folder="logs/",
    )
    # speakers = audio.SpeakerModule()
    amq = TextAnswertoBEATBridge()

    mic.subscribe(vad)
    vad.subscribe(asr)
    asr.subscribe(amq)

    # running system
    try:
        network.run(mic)
        logger.info("Dialog system ready")
        wait("q")
        network.stop(mic)
    except Exception:
        logger.exception("test")
        # network.stop(mic)


if __name__ == "__main__":
    logger = structlog.get_logger("basic logger")

    try:
        # test_structlog()
        txt = "lala"
        pred = "lala lala"
        # textiu = TextIU()
        # speechreciu = SpeechRecognitionIU(None)
        # print(textiu)
        # print(speechreciu)
        # textiu.payload = txt
        # speechreciu.payload = txt
        # print(textiu)
        # print(speechreciu)

        logger = structlog.get_logger()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        printing = False
        frame_length = 0.02
        rate = 16000
        asr = WhisperASRInterruptionModule(
            device=device,
            printing=False,
            full_sentences=True,
            input_framerate=16000,
            log_folder="logs/",
        )

        asriu = asr.create_iu()
        print(asriu)
        asriu2 = asr.create_iu(
            grounded_in=None,
            predictions=[pred],
            text=txt,
            stability=0.0,
            confidence=0.99,
            final=False,
        )
        print(asriu2)
        print(asriu2.__dict__)
        print(asriu2.created_at)
        print(type(asriu2.created_at))
        print(asriu2.created_at.isoformat())
    except Exception:
        logger.exception("error :")
