import json
import datetime
import os
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib


def extract_number(f):
    s = re.findall("\d+$", f)
    return (int(s[0]) if s else -1, f)


def plotting_run(logfile_path=None, plot_saving_path=None):
    if logfile_path is None or plot_saving_path is None:
        subfolders = [f.path for f in os.scandir("logs/") if f.is_dir()]
        max_run = max(subfolders, key=extract_number)
        logfile_path = max_run + "/logs.log"
        plot_saving_path = "screens/" + max_run.split("/")[-1]
    x_axis = []
    y_axis = []
    y_axis_append_UM = []
    x_axis_append_UM = []
    x_axis_process_update = []
    y_axis_process_update = []
    nb_pb_line = 0
    with open(logfile_path, encoding="utf-8") as f:
        lines = f.readlines()

        for i, l in enumerate(lines):
            pb_line = i, l
            try:
                log = json.loads(l)
                date = datetime.datetime.fromisoformat(log["timestamp"])
                date_plt = mdates.date2num(date)
                module_name = " ".join(log["module"].split()[:1])
                if log["event"] == "create_iu":
                    x_axis.append(date_plt)
                    y_axis.append(module_name)
                elif log["event"] == "append UM":
                    x_axis_append_UM.append(date_plt)
                    y_axis_append_UM.append(module_name)
                elif log["event"] == "process_update":
                    x_axis_process_update.append(date_plt)
                    y_axis_process_update.append(module_name)
            except Exception:
                nb_pb_line += 1

    print("nb_pb_line = ", nb_pb_line)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_axis, y_axis, "+", color="b", label="create_iu", markersize=7)
    ax.plot(
        x_axis_append_UM,
        y_axis_append_UM,
        "^",
        color="c",
        label="append UM",
        markersize=3,
    )
    ax.plot(
        x_axis_process_update,
        y_axis_process_update,
        "o",
        color="darkorange",
        label="process_update",
        markersize=1,
    )

    ax.grid(True)
    ax.legend()
    plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M:%S.%f"))
    plt.xticks(fontsize=7)

    # saving the plot
    if not os.path.isdir(plot_saving_path):
        os.makedirs(plot_saving_path)
    plot_filename = plot_saving_path + "/plot_IU_exchange.png"
    # plt.figsize = (10, 3)
    plt.savefig(plot_filename, dpi=200, bbox_inches="tight")

    # showing the plot
    # plt.show()


def get_latency(logfile_path=None):
    if logfile_path is None:
        subfolders = [f.path for f in os.scandir("logs/") if f.is_dir()]
        max_run = max(subfolders, key=extract_number)
        logfile_path = max_run + "/logs.log"
    nb_pb_line = 0

    # get all append UM messages per module
    with open(logfile_path, encoding="utf-8") as f:
        lines = f.readlines()
        messages_per_module = {}
        for i, l in enumerate(lines):
            pb_line = i, l
            try:
                log = json.loads(l)
                if log["event"] == "append UM":
                    date = datetime.datetime.fromisoformat(log["timestamp"])
                    if log["module"] not in messages_per_module:
                        messages_per_module[log["module"]] = [date]
                    else:
                        messages_per_module[log["module"]].append(date)
            except Exception:
                nb_pb_line += 1

    # filter to find the first and last append UM of each turn
    first_and_last_per_module = {}
    for module, dates in messages_per_module.items():
        first_and_last_per_module[module] = [[], []]
        # last append UM per turn per module
        for i, message_date in enumerate(dates):
            if i != len(dates) - 1:
                # print(f"dates : {message_date} , {dates[i + 1]}")
                # print(f"dates sub : {dates[i + 1] - message_date}")
                # print(f"dates sub : {(dates[i + 1] - message_date).total_seconds()}")
                if (dates[i + 1] - message_date).total_seconds() > 1.2:
                    first_and_last_per_module[module][1].append(message_date)
            else:
                first_and_last_per_module[module][1].append(message_date)
        # first append UM per turn per module
        for i, message_date in enumerate(dates):
            if i != 0:
                # print(f"dates : {message_date} , {dates[i - 1]}")
                # print(f"dates sub : {message_date - dates[i - 1]}")
                # print(f"dates sub : {(message_date - dates[i - 1]).total_seconds()}")
                if (message_date - dates[i - 1]).total_seconds() > 1.2:
                    first_and_last_per_module[module][0].append(message_date)
            else:
                first_and_last_per_module[module][0].append(message_date)

    print(
        f"first_and_last_per_module = {[(len(v[0]), len(v[1])) for k,v in first_and_last_per_module.items()]}",
    )
    # print("first_and_last_per_module = ", first_and_last_per_module)
    modules = [(key, value) for key, value in first_and_last_per_module.items()]
    last_module = modules[0][0]
    modules = modules[1:]
    for module, dates in modules:
        firsts = dates[0]
        for i, fdate in enumerate(firsts):
            # get last date from previous module in pipeline
            previous_dates_from_previous_module = [
                date
                for date in first_and_last_per_module[last_module][0]
                if date < fdate
            ]

            latency = (fdate - max(previous_dates_from_previous_module)).total_seconds()
            print(f"latency {module}, {latency}")

        last_module = module


import pandas as pd


def test_pandas(logfile_path=None):
    if logfile_path is None:
        subfolders = [f.path for f in os.scandir("logs/") if f.is_dir()]
        max_run = max(subfolders, key=extract_number)
        logfile_path = max_run + "/logs.log"
    with open(logfile_path, encoding="utf-8") as f:
        lines = f.readlines()
        lines_json = []
        for i, l in enumerate(lines):
            try:
                lines_json.append(json.loads(l))
            except Exception:
                pass
    df = pd.DataFrame(lines_json)
    print(df)
    print(df[df["module"] == "Microphone Module"])


def test_pandas(logfile_path=None):
    if logfile_path is None:
        subfolders = [f.path for f in os.scandir("logs/") if f.is_dir()]
        max_run = max(subfolders, key=extract_number)
        logfile_path = max_run + "/logs.log"
    with open(logfile_path, encoding="utf-8") as f:
        lines = f.readlines()
        lines_json = []
        for i, l in enumerate(lines):
            try:
                lines_json.append(json.loads(l))
            except Exception:
                pass

    df = pd.DataFrame(lines_json)

    # GET ALL TIMESTAMP FOR SPEAKERS APPEND MSG (EOT timestamps)
    spk_msgs = df[df["module"] == "Speaker Interruption Module"]
    append_spk_msgs = spk_msgs[spk_msgs["event"] == "append UM"]
    EOT_timestamps = append_spk_msgs["timestamp"]
    EOT_timestamps = EOT_timestamps.drop_duplicates()
    print(EOT_timestamps)

    # GET ALL TIMESTAMP FOR VAD APPEND MSG (BOT timestamps)
    vad_msgs = df[df["module"] == "VADTurn Module"]
    append_vad_msgs = vad_msgs[vad_msgs["event"] == "append UM"]
    BOT_timestamps = append_vad_msgs["timestamp"]
    BOT_timestamps = BOT_timestamps.drop_duplicates()
    print(BOT_timestamps)


if __name__ == "__main__":

    # test_structlog()
    # test_plot()
    # logfile_path = "logs/run_1/logs.log"
    # plot_saving_path = "screens/run_1"
    # plotting_run(logfile_path, plot_saving_path)

    # plotting_run()

    # get_latency()

    test_pandas()
