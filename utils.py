import csv
import datetime
import os


# LOGS FUNCTIONS
def manage_log_folder(log_folder, file_name):
    complete_path = log_folder + "/" + file_name
    if os.path.isfile(complete_path):  # if file path already exists
        return create_new_log_folder(log_folder) + "/" + file_name
    else:
        print("log_folder = ", log_folder)
        if not os.path.isdir(log_folder):
            os.makedirs(log_folder)
        return log_folder + "/" + file_name


def create_new_log_folder(log_folder):
    cpt = 0
    filename = log_folder + "_" + str(cpt)
    while os.path.isdir(filename):
        cpt += 1
        filename = log_folder + "_" + str(cpt)
    os.mkdir(filename)
    print("create dir = ", filename)
    return filename


def write_logs(log_file, rows):
    with open(log_file, "a", newline="") as f:
        csv_writer = csv.writer(f)
        for row in rows:
            csv_writer.writerow(row)


def merge_logs(log_folder):
    wozmic_file = log_folder + "/wozmic.csv"
    asr_file = log_folder + "/asr.csv"
    llm_file = log_folder + "/llm.csv"
    tts_file = log_folder + "/tts.csv"
    speaker_file = log_folder + "/speaker.csv"
    files = [wozmic_file, asr_file, llm_file, tts_file, speaker_file]

    res_file = log_folder + "/res.csv"
    # res_file = manage_log_folder(log_folder, "res.csv")

    date_format = "%H:%M:%S.%f"

    with open(res_file, "w", newline="") as w:
        writer = csv.writer(w)
        writer.writerow(["Module", "Start", "Stop", "Duration"])
        first_start = None
        last_stop = None
        for fn in files:
            # print(fn)
            if os.path.isfile(fn):
                with open(fn, "r") as f:
                    l = [fn, 0, 0, 0]
                    for row in csv.reader(
                        f
                    ):  # TODO : is there only 1 start and 1 stop ?
                        if row[0] == "Start":
                            l[1] = row[1]
                            if first_start is None or first_start > l[1]:
                                first_start = l[1]
                        elif row[0] == "Stop":
                            l[2] = row[1]
                            if last_stop is None or last_stop < l[1]:
                                last_stop = l[1]
                    if l[1] != 0 and l[2] != 0:
                        l[3] = datetime.datetime.strptime(
                            l[2], date_format
                        ) - datetime.datetime.strptime(l[1], date_format)
                    writer.writerow(l)

        total_duration = datetime.datetime.strptime(
            last_stop, date_format
        ) - datetime.datetime.strptime(first_start, date_format)
        writer.writerow(["Total", first_start, last_stop, total_duration])
