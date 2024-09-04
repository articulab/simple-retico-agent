from keyboard import wait
from retico_core import *
import structlog


def test_structlog():
    logger = structlog.get_logger()

    mic = audio.MicrophoneModule()
    speakers = audio.SpeakerModule()
    mic.subscribe(speakers)

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
    test_structlog()
