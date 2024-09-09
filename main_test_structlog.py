from keyboard import wait
import matplotlib
from retico_core import *
from retico_core.abstract import *
from retico_core.text import *
from retico_core.utils import *
import structlog
import torch

from amq import TextAnswertoBEATBridge
from vad_turn import VADTurnModule
from whisper_asr_interruption import WhisperASRInterruptionModule


def test_structlog():
    logger = structlog.get_logger()
    # log_folder = create_new_log_folder("logs/run")
    log_folder = "logs/run"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    printing = False
    frame_length = 0.02
    rate = 16000

    mic = audio.MicrophoneModule()
    asr = WhisperASRInterruptionModule(
        device=device,
        printing=False,
        full_sentences=True,
        input_framerate=16000,
        # log_folder=log_folder,
    )
    vad = VADTurnModule(
        printing=printing,
        input_framerate=rate,
        # log_folder=log_folder,
        frame_length=frame_length,
    )

    # speakers = audio.SpeakerModule()
    amq = TextAnswertoBEATBridge()

    mic.subscribe(vad)
    vad.subscribe(asr)
    asr.subscribe(amq)

    # running system
    try:
        network.run(mic, log_folder)
        logger.info("Dialog system ready")
        wait("q")
        network.stop(mic)
    except Exception:
        logger.exception("test")
        # network.stop(mic)


import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def test_plot():
    logger = structlog.get_logger()
    file_path = "logs/run_10/logs.log"
    x_axis = []
    y_axis = []
    x_axis2 = []
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()
        # print(lines[1])
        line1 = lines[1]
        line1json = json.loads(line1)
        logger.info(line1json["created_at"])
        logger.info(datetime.datetime.fromisoformat(line1json["created_at"]))
        for l in lines:
            log = json.loads(l)
            date = datetime.datetime.fromisoformat(log["created_at"])
            date_plt = mdates.date2num(date)
            x_axis.append(date_plt)
            x_axis2.append(date)
            y_axis.append(log["module"])

    fig, ax = plt.subplots()
    # ax.plot(x_axis, y_axis, "x")
    ax.plot(x_axis2, y_axis, ".")
    # ax.xaxis.set_tick_params(rotation=40)
    plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M:%S.%f"))
    plt.show()

    # dates = matplotlib.dates.datestr2num(x_axis2)  # convert string dates to numbers
    # plt.plot_date(dates, y_axis)  # doesn't show milliseconds by default.

    # # This changes the formatter.
    # plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M:%S.%f"))

    # # Redraw the plot.
    # plt.draw()

    # date = datetime.datetime.now()
    # date_str = str(date)
    # date_as_str = "datetime.datetime(2024, 9, 9, 15, 8, 24, 359757)"
    # date_iso = date.isoformat()
    # logger.info(date)
    # logger.info(date_str)
    # logger.info(date_as_str)
    # logger.info(date_iso)
    # print(date)

    # date_str_rev = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f")
    # # date_as_str_rev = datetime.datetime.fromisoformat(date_iso)
    # date_iso_rev = datetime.datetime.fromisoformat(date_iso)
    # logger.info(date_str_rev)
    # # logger.info(date_str)
    # logger.info(date_iso_rev)


if __name__ == "__main__":

    test_structlog()
    # test_plot()
