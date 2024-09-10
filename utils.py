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
                    module_name = " ".join(log["module"].split()[:1])
                    y_axis.append(module_name)
                elif log["event"] == "append UM":
                    module_name = " ".join(log["module"].split()[:1])
                    y_axis_append_UM.append(module_name)
                    x_axis_append_UM.append(date_plt)
            except Exception:
                nb_pb_line += 1

    print("nb_pb_line = ", nb_pb_line)

    # plt.figure()
    fig, ax = plt.subplots(figsize=(10, 5))
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
    plt.xticks(fontsize=7)

    # saving the plot
    if not os.path.isdir(plot_saving_path):
        os.makedirs(plot_saving_path)
    plot_filename = plot_saving_path + "/plot_IU_exchange.png"
    # plt.figsize = (10, 3)
    plt.savefig(plot_filename, dpi=200, bbox_inches="tight")

    # showing the plot
    # plt.show()


if __name__ == "__main__":

    # test_structlog()
    # test_plot()
    # logfile_path = "logs/run_1/logs.log"
    # plot_saving_path = "screens/run_1"
    # plotting_run(logfile_path, plot_saving_path)

    plotting_run()
