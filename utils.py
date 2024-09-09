import csv
import datetime
import os

import torch
import retico_core


class TextAlignedAudioIU(retico_core.audio.AudioIU):
    """AudioIU enhanced with information that aligns the AudioIU to the current written agent turn.

    Attributes:
        - grounded_word : the word corresponding to the audio.
        - turn_id (int) : The index of the dialogue's turn the IU is part of.
        - clause_id (int) : The index of the clause the IU is part of, in the current turn.
        - word_id (int) : The index of the word that corresponds to the end of the IU].
        - char_id (int) : The index of the last character from the grounded_word.
        - final (bool) : Wether the IU is an EOT.
    """

    @staticmethod
    def type():
        return "Text Aligned Audio IU"

    def __init__(
        self,
        creator=None,
        iuid=0,
        previous_iu=None,
        grounded_in=None,
        audio=None,
        rate=None,
        nframes=None,
        sample_width=None,
        grounded_word=None,
        word_id=None,
        char_id=None,
        turn_id=None,
        clause_id=None,
        final=None,
        **kwargs,
    ):
        super().__init__(
            creator=creator,
            iuid=iuid,
            previous_iu=previous_iu,
            grounded_in=grounded_in,
            payload=audio,
            raw_audio=audio,
            rate=rate,
            nframes=nframes,
            sample_width=sample_width,
        )
        self.grounded_word = grounded_word
        self.word_id = word_id
        self.char_id = char_id
        self.turn_id = turn_id
        self.clause_id = clause_id
        self.final = final

    def set_data(
        self,
        grounded_word=None,
        word_id=None,
        char_id=None,
        turn_id=None,
        clause_id=None,
        audio=None,
        chunk_size=None,
        rate=None,
        sample_width=None,
        final=False,
    ):
        """Sets AudioIU parameters and the alignment information"""
        # alignment information
        self.grounded_word = grounded_word
        self.word_id = word_id
        self.char_id = char_id
        self.turn_id = turn_id
        self.clause_id = clause_id
        self.final = final
        # AudioIU information
        self.payload = audio
        self.raw_audio = audio
        self.rate = rate
        self.nframes = chunk_size
        self.sample_width = sample_width


class TurnTextIU(retico_core.text.TextIU):
    """TextIU enhanced with information related to dialogue turns, clauses, etc.

    Attributes:
        - turn_id (int) : Which dialogue's turn the IU is part of.
        - clause_id (int) : Which clause the IU is part of, in the current turn.
        - final (bool) : Wether the IU is an EOT.
    """

    @staticmethod
    def type():
        return "Turn Text IU"

    def __init__(
        self,
        creator=None,
        iuid=0,
        previous_iu=None,
        grounded_in=None,
        text=None,
        turn_id=None,
        clause_id=None,
        final=False,
        **kwargs,
    ):
        super().__init__(
            creator=creator,
            iuid=iuid,
            previous_iu=previous_iu,
            grounded_in=grounded_in,
            payload=text,
        )
        self.turn_id = turn_id
        self.clause_id = clause_id
        self.final = final

    def set_data(
        self,
        text=None,
        turn_id=None,
        clause_id=None,
        final=False,
    ):
        """Sets TextIU parameters and dialogue turns informations (turn_id, clause_id, final)"""
        # dialogue turns information
        self.turn_id = turn_id
        self.clause_id = clause_id
        self.final = final
        # TextIU information
        self.payload = text
        self.text = text


class VADTurnAudioIU(retico_core.audio.AudioIU):
    """AudioIU enhanced by VADTurnModule with dialogue turn information (agent_turn, user_turn,
    silence, interruption, etc) contained in the vad_state parameter.

    Attributes:
        vad_state (string): dialogue turn information (agent_turn, user_turn, silence, interruption, etc) from VADTurnModule.
    """

    @staticmethod
    def type():
        return "VADTurn Audio IU"

    def __init__(
        self,
        creator=None,
        iuid=0,
        previous_iu=None,
        grounded_in=None,
        audio=None,
        vad_state=None,
        rate=None,
        nframes=None,
        sample_width=None,
        **kwargs,
    ):
        super().__init__(
            creator=creator,
            iuid=iuid,
            previous_iu=previous_iu,
            grounded_in=grounded_in,
            payload=audio,
            raw_audio=audio,
            rate=rate,
            nframes=nframes,
            sample_width=sample_width,
        )
        self.vad_state = vad_state

    def set_data(
        self, vad_state=None, audio=None, chunk_size=None, rate=None, sample_width=None
    ):
        """Sets AudioIU parameters and vad_state"""
        # vad_state
        self.vad_state = vad_state
        # AudioIU information
        self.payload = audio
        self.raw_audio = audio
        self.rate = rate
        self.nframes = chunk_size
        self.sample_width = sample_width


# # LOGS FUNCTIONS
# def manage_log_folder(log_folder, file_name):
#     complete_path = log_folder + "/" + file_name
#     if os.path.isfile(complete_path):  # if file path already exists
#         return create_new_log_folder(log_folder) + "/" + file_name
#     else:
#         print("log_folder = ", log_folder)
#         if not os.path.isdir(log_folder):
#             os.makedirs(log_folder)
#         return log_folder + "/" + file_name


# def create_new_log_folder(log_folder):
#     cpt = 0
#     filename = log_folder + "_" + str(cpt)
#     while os.path.isdir(filename):
#         cpt += 1
#         filename = log_folder + "_" + str(cpt)
#     os.mkdir(filename)
#     print("create dir = ", filename)
#     return filename


# def write_logs(log_file, rows):
#     with open(log_file, "a", newline="") as f:
#         csv_writer = csv.writer(f)
#         for row in rows:
#             csv_writer.writerow(row)


# def merge_logs(log_folder):
#     wozmic_file = log_folder + "/wozmic.csv"
#     if not os.path.isfile(wozmic_file):
#         return None
#     asr_file = log_folder + "/asr.csv"
#     llm_file = log_folder + "/llm.csv"
#     tts_file = log_folder + "/tts.csv"
#     speaker_file = log_folder + "/speaker.csv"
#     files = [wozmic_file, asr_file, llm_file, tts_file, speaker_file]

#     res_file = log_folder + "/res.csv"
#     date_format = "%H:%M:%S.%f"

#     with open(res_file, "w", newline="") as w:
#         writer = csv.writer(w)
#         writer.writerow(["Module", "Start", "Stop", "Duration"])
#         first_start = None
#         last_stop = None
#         for fn in files:
#             if os.path.isfile(fn):
#                 with open(fn, "r") as f:
#                     l = [fn, None, None, 0]
#                     for row in csv.reader(
#                         f
#                     ):  # TODO : is there only 1 start and 1 stop ?
#                         if row[0] == "Start":
#                             if l[1] is None or l[1] > row[1]:
#                                 l[1] = row[1]
#                             if first_start is None or first_start > row[1]:
#                                 first_start = row[1]
#                         elif row[0] == "Stop":
#                             if l[2] is None or l[2] < row[1]:
#                                 l[2] = row[1]
#                             if last_stop is None or last_stop < row[1]:
#                                 last_stop = row[1]
#                     if l[1] is not None and l[2] is not None:
#                         l[3] = datetime.datetime.strptime(
#                             l[2], date_format
#                         ) - datetime.datetime.strptime(l[1], date_format)
#                     writer.writerow(l)

#         total_duration = datetime.datetime.strptime(
#             last_stop, date_format
#         ) - datetime.datetime.strptime(first_start, date_format)
#         writer.writerow(["Total", first_start, last_stop, total_duration])


# # DEVICE DEF
# def device_definition(device):
#     cuda_available = torch.cuda.is_available()
#     final_device = None
#     if device is None:
#         if cuda_available:
#             final_device = "cuda"
#         else:
#             final_device = "cpu"
#     elif device == "cuda":
#         if cuda_available:
#             final_device = "cuda"
#         else:
#             print(
#                 "device defined for instantiation is cuda but cuda is not available. Check you cuda installation if you want the module to run on GPU. The module will run on CPU instead."
#             )
#             # Raise Exception("device defined for instantiation is cuda but cuda is not available. check you cuda installation or change device to "cpu")
#             final_device = "cpu"
#     elif device == "cpu":
#         if cuda_available:
#             print(
#                 "cuda is available, you can run the module on GPU by changing the device parameter to cuda."
#             )
#         final_device = "cpu"
#     return final_device
