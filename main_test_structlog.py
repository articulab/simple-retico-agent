from keyboard import wait
import matplotlib
from retico_core import *
from retico_core.abstract import *
from retico_core.text import *
from retico_core.log_utils import *
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
    file_path = "logs/run_1/logs.log"
    x_axis = []
    x_axis2 = []
    y_axis = []
    y_axis_append_UM = []
    x_axis_append_UM = []
    pb_line = None
    nb_pb_line = 0
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

        for i, l in enumerate(lines):
            pb_line = i, l
            try:
                log = json.loads(l)
                date = datetime.datetime.fromisoformat(log["timestamp"])
                date_plt = mdates.date2num(date)
                if log["event"] == "create_iu":
                    x_axis.append(date_plt)
                    x_axis2.append(date)
                    y_axis.append(log["module"])
                elif log["event"] == "append UM":
                    y_axis_append_UM.append(log["module"])
                    x_axis_append_UM.append(date_plt)
            except Exception:
                nb_pb_line += 1

    print("nb_pb_line = ", nb_pb_line)

    fig, ax = plt.subplots()
    # ax.plot(x_axis, y_axis, "x")
    # ax.plot(x_axis2, y_axis, ".")
    ax.plot(
        x_axis_append_UM,
        y_axis_append_UM,
        "^",
        color="r",
        label="append UM",
        markersize=5,
    )
    ax.plot(x_axis, y_axis, ".", color="b", label="create_iu", markersize=3)

    ax.grid(True)
    ax.legend()
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


def plotting_run(logfile_path, plot_saving_path):
    logger = structlog.get_logger()
    logfile_path = "logs/run_1/logs.log"
    x_axis = []
    y_axis = []
    y_axis_append_UM = []
    x_axis_append_UM = []
    nb_pb_line = 0
    with open(logfile_path, encoding="utf-8") as f:
        lines = f.readlines()

        for i, l in enumerate(lines):
            pb_line = i, l
            try:
                log = json.loads(l)
                date = datetime.datetime.fromisoformat(log["timestamp"])
                date_plt = mdates.date2num(date)
                if log["event"] == "create_iu":
                    x_axis.append(date_plt)
                    y_axis.append(log["module"])
                elif log["event"] == "append UM":
                    y_axis_append_UM.append(log["module"])
                    x_axis_append_UM.append(date_plt)
            except Exception:
                nb_pb_line += 1

    print("nb_pb_line = ", nb_pb_line)

    fig, ax = plt.subplots()
    ax.plot(
        x_axis_append_UM,
        y_axis_append_UM,
        "^",
        color="r",
        label="append UM",
        markersize=5,
    )
    ax.plot(x_axis, y_axis, ".", color="b", label="create_iu", markersize=3)

    ax.grid(True)
    ax.legend()
    plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M:%S.%f"))

    # saving the plot
    if not os.path.isdir(plot_saving_path):
        os.makedirs(plot_saving_path)
    plot_filename = plot_saving_path + "/plot_IU_exchange.png"
    plt.savefig(plot_filename)

    # showing the plot
    plt.show()


if __name__ == "__main__":

    # test_structlog()
    # test_plot()
    logfile_path = "logs/run_1/logs.log"
    plot_saving_path = "screens/run_1"

    plotting_run(logfile_path, plot_saving_path)
