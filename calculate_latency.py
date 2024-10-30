import datetime
import json
import os

import pandas as pd

from retico_core.log_utils import extract_number


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


def test_pandas(logfile_path=None):
    if logfile_path is None:
        subfolders = [f.path for f in os.scandir("logs/") if f.is_dir()]
        max_run = max(subfolders, key=extract_number)
        logfile_path = max_run + "/logs.log"
    with open(logfile_path, encoding="utf-8") as f:
        lines = f.readlines()
        lines_json = []
        for _, l in enumerate(lines):
            try:
                lines_json.append(json.loads(l))
            except Exception:
                pass
    df = pd.DataFrame(lines_json)
    print(df)
    print(df[df["module"] == "Microphone Module"])


def pandas_latency(logfile_path=None):
    if logfile_path is None:
        subfolders = [f.path for f in os.scandir("logs/") if f.is_dir()]
        max_run = max(subfolders, key=extract_number)
        logfile_path = max_run + "/logs.log"
    with open(logfile_path, encoding="utf-8") as f:
        lines = f.readlines()
        lines_json = []
        for i, l in enumerate(lines):
            try:
                line = json.loads(l)
                line["timestamp"] = datetime.datetime.fromisoformat(line["timestamp"])
                lines_json.append(line)
            except Exception:
                pass

    df = pd.DataFrame(lines_json)

    # GET ALL TIMESTAMP FOR SPEAKERS APPEND MSG (EOT timestamps)
    spk_msgs = df[df["module"] == "Speaker Interruption Module"]
    append_spk_msgs = spk_msgs[spk_msgs["event"] == "append UM"]
    append_spk_msgs = append_spk_msgs.drop_duplicates(subset=["timestamp"])
    print(append_spk_msgs)
    EOT_timestamps = append_spk_msgs["timestamp"]
    print(EOT_timestamps)

    # GET ALL TIMESTAMP FOR VAD APPEND MSG (BOT timestamps)
    vad_msgs = df[df["module"] == "VADTurn Module"]
    append_vad_msgs = vad_msgs[vad_msgs["event"] == "append UM"]
    BOT_append_vad_msgs = append_vad_msgs[
        append_vad_msgs["timestamp"].diff().gt(pd.Timedelta("0.1 s"))
    ]
    first_BOT = append_vad_msgs.iloc[[0]]
    BOTs = pd.concat([first_BOT, BOT_append_vad_msgs])
    print(BOTs)
    BOT_timestamps = BOTs["timestamp"]
    print(BOT_timestamps)

    # trying to have turns as tuple (BOT_timestamps, EOT_timestapms) but not easy iwith pandas.
    # BOT_timestamps_f = BOT_timestamps.to_frame()
    # EOT_timestamps_f = EOT_timestamps.to_frame()
    # print(BOT_timestamps_f)
    # print(EOT_timestamps_f)
    # print(EOT_timestamps_f.columns.intersection(EOT_timestamps_f.columns))
    # Turns_timestamps = BOT_timestamps_f.join(
    #     EOT_timestamps, lsuffix="_BOT", rsuffix="_EOT"
    # )
    # print(Turns_timestamps)

    # check that BOT and EOT have same length
    assert BOT_timestamps.size == EOT_timestamps.size

    # get each module's first append UM of each turn
    append_msgs = df[df["event"] == "append UM"]
    for turn_id in range(BOT_timestamps.size):
        print("Turn ", turn_id)
        bot = BOT_timestamps.iloc[turn_id]
        eot = EOT_timestamps.iloc[turn_id]
        # print(BOT)
        # print(EOT)
        # print(append_msgs["timestamp"])
        append_msgs_turn_i = append_msgs[
            (append_msgs["timestamp"] >= bot) & (append_msgs["timestamp"] <= eot)
        ]
        # print(append_msgs_turn_i["timestamp"])

        #
        print("first and last append Um of each module")
        turn_i_per_module = append_msgs_turn_i.groupby("module")
        turn_i_per_module_first = turn_i_per_module["timestamp"].min()
        turn_i_per_module_last = turn_i_per_module["timestamp"].max()
        # print(turn_i_per_module)
        print(turn_i_per_module_first)
        print(turn_i_per_module_last)

        # TODO: calculate the latency between each module's first append UM message
        pipeline = [
            "Microphone Module",
            "VADTurn Module",
            "WhisperASR Module",
            "LlamaCppMemoryIncremental Module",
            "CoquiTTS Module",
            "Speaker Interruption Module",
        ]
        for i, module in enumerate(pipeline[1:]):
            piepline_diff = (
                turn_i_per_module_first[module]
                - turn_i_per_module_first[pipeline[i - 1]]
            )
            print(piepline_diff)


def pandas_latency_2(logfile_path=None):
    if logfile_path is None:
        subfolders = [f.path for f in os.scandir("logs/") if f.is_dir()]
        max_run = max(subfolders, key=extract_number)
        logfile_path = max_run + "/logs.log"
    with open(logfile_path, encoding="utf-8") as f:
        lines = f.readlines()
        lines_json = []
        for i, l in enumerate(lines):
            try:
                line = json.loads(l)
                line["timestamp"] = datetime.datetime.fromisoformat(line["timestamp"])
                lines_json.append(line)
            except Exception:
                pass

    df = pd.DataFrame(lines_json)

    # Get user BOTs & EOTs
    user_BOTs = df[df["event"] == "user_BOT"]  # 12, 2, div/6
    user_EOTs = df[df["event"] == "user_EOT"]  # 30, 5, div/6
    agent_BOTs = user_EOTs  # 30, 5, div/6
    agent_EOTs = df[df["event"] == "agent_EOT"]  # 12, 2, div/6
    interruptions = df[df["event"] == "interruption"]  # 36, 6, div/6
    user_BARGE_IN = interruptions[
        interruptions["module"] == "VADTurn Module"
    ]  # 18, 3, div/6
    user_TOT_BOTs = pd.concat([user_BOTs, user_BARGE_IN])  # 30, 5, div/6
    agent_TOT_EOTs = pd.concat([agent_EOTs, user_BARGE_IN])  # 30, 5, div/6
    user_TOT_BOTs = user_TOT_BOTs.sort_values(by=["timestamp"])
    agent_TOT_EOTs = agent_TOT_EOTs.sort_values(by=["timestamp"])
    user_BOTs = user_BOTs.drop_duplicates()
    user_EOTs = user_EOTs.drop_duplicates()
    agent_BOTs = agent_BOTs.drop_duplicates()
    agent_EOTs = agent_EOTs.drop_duplicates()
    interruptions = interruptions.drop_duplicates()
    user_BARGE_IN = user_BARGE_IN.drop_duplicates()
    user_TOT_BOTs = user_TOT_BOTs.drop_duplicates()
    agent_TOT_EOTs = agent_TOT_EOTs.drop_duplicates()
    speaker_outputs = df[df["event"] == "output_audio"].drop_duplicates()

    append_msgs = df[df["event"] == "append UM"]
    pipeline = [
        "Microphone Module",
        "VADTurn Module",
        "WhisperASR Module",
        "LlamaCppMemoryIncremental Module",
        "CoquiTTS Module",
        "Speaker Interruption Module",
    ]
    for turn_id in range(user_TOT_BOTs["timestamp"].size):

        try:
            #######
            # Calculate the user and agent BOT and EOT and the timestamp of the first agent audio output
            ######
            user_BOT = user_TOT_BOTs.iloc[turn_id]["timestamp"]
            if turn_id < user_TOT_BOTs["timestamp"].size - 1:
                user_EOT = user_EOTs[
                    (user_EOTs["timestamp"] >= user_BOT)
                    & (
                        user_EOTs["timestamp"]
                        <= user_TOT_BOTs.iloc[turn_id + 1]["timestamp"]
                    )
                ]
                user_EOT = max(user_EOT["timestamp"])
                agent_EOT = agent_TOT_EOTs[
                    (agent_TOT_EOTs["timestamp"] >= user_BOT)
                    & (
                        agent_TOT_EOTs["timestamp"]
                        <= user_TOT_BOTs.iloc[turn_id + 1]["timestamp"]
                    )
                ]
                agent_EOT = max(agent_EOT["timestamp"])
                speaker_first_output = speaker_outputs[
                    (speaker_outputs["timestamp"] >= user_BOT)
                    & (
                        speaker_outputs["timestamp"]
                        <= user_TOT_BOTs.iloc[turn_id + 1]["timestamp"]
                    )
                ]
                speaker_first_output = min(speaker_first_output["timestamp"])
            else:
                speaker_first_output = speaker_outputs[
                    speaker_outputs["timestamp"] >= user_BOT
                ]
                speaker_first_output = min(speaker_first_output["timestamp"])
                user_EOT = max(user_EOTs["timestamp"])
                agent_EOT = max(agent_TOT_EOTs["timestamp"])

            print(f"\n\nTURN {turn_id} : ")
            print(f"{'user BOT :'.ljust(25, ' ')} {user_BOT}")
            print(f"{'user EOT / agent BOT :'.ljust(25, ' ')} {user_EOT}")
            print(f"{'agent first output :'.ljust(25, ' ')} {speaker_first_output}")
            print(f"{'agent EOT :'.ljust(25, ' ')} {agent_EOT}\n")

            #######
            # Print the timestamp of the first data output for each module
            #######
            if turn_id < user_TOT_BOTs["timestamp"].size - 1:
                append_msgs_turn = append_msgs[
                    (append_msgs["timestamp"] >= user_EOT)
                    & (append_msgs["timestamp"] <= agent_EOT)
                ]
            else:
                append_msgs_turn = append_msgs[append_msgs["timestamp"] >= user_EOT]
            append_msgs_turn_module = append_msgs_turn.groupby("module")
            append_msgs_turn_module_BOT = append_msgs_turn_module["timestamp"].min()
            append_msgs_turn_module_BOT = (
                append_msgs_turn_module_BOT.to_frame().sort_values(by="timestamp")
            )

            # Changing the values for Microphone Module and Speaker Module because they are logged differently
            append_msgs_turn_module_BOT.loc[
                "Speaker Interruption Module", "timestamp"
            ] = speaker_first_output
            append_msgs_turn_module_BOT.loc["Microphone Module", "timestamp"] = user_EOT
            print(f"Timestamp first data append : \n{append_msgs_turn_module_BOT}\n")

            append_msgs_turn_module_BOT = append_msgs_turn_module_BOT

            #######
            # Calculate the EXEC TIME of each module at each turn of the dialogue
            # (EXEC TIME = the time for each module between receiving the first data and outputting the first data)
            # the sum of the EXEC TIME of all modules = the answering latency of the system
            ######
            total_exec_time = pd.Timedelta(0)
            start_id = 1
            for i, module in enumerate(pipeline[start_id:]):
                pipeline_diff = (
                    append_msgs_turn_module_BOT.loc[module, "timestamp"]
                    - append_msgs_turn_module_BOT.loc[
                        pipeline[i + start_id - 1], "timestamp"
                    ]
                )
                total_exec_time += pipeline_diff
                print(f"EXEC TIME   {module.ljust(40, ' ')} {pipeline_diff}")

            print(f"\nTOTAL EXEC TIME   {str(total_exec_time).ljust(40, ' ')}")
            print(
                f"TOTAL EXEC TIME   {str(speaker_first_output-user_EOT).ljust(40, ' ')}"
            )

        except ValueError:
            print(
                "Turns where the user interrupted the agent before the agent had time to speak are not interesting for now"
            )


if __name__ == "__main__":
    # get_latency()
    # pandas_latency()
    pandas_latency_2()
