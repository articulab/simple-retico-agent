from functools import partial
import logging
import structlog

from retico_core.log_utils import filter_value_not_in_list, filter_cases, log_exception

# LOG_FILTERS = []

# LOG_FILTERS = [
#     partial(
#         filter_cases,
#         cases=[
#             [
#                 ("level", ["error", "exception"]),
#             ],
#             [
#                 (
#                     "event",
#                     ["thread exception", "error", "exception"],
#                 ),
#             ],
#             [
#                 ("exc_info", [True]),
#             ],
#         ],
#     ),
# ]

# # LOG_FILTERS = [
# #     # partial(filter_does_not_have_key, key="module"),
# #     # partial(
# #     #     filter_value_in_list,
# #     #     key="module",
# #     #     values=["Microphone Module", "VADTurn Module"],
# #     # ),
# #     # partial(filter_has_key, key="module"),
# #     # partial(
# #     #     filter_value_not_in_list,
# #     #     key="level",
# #     #     values=["error"],
# #     # ),
# #     partial(
# #         filter_value_not_in_list,
# #         key="event",
# #         values=[
# #             "new iu",
# #             "sent",
# #             "error",
# #             "before sent",
# #             "AMQWriter sends a message to ActiveMQ",
# #             "AMQReader receives a message from ActiveMQ",
# #             "AMQReader creates new iu",
# #             "CallbackModule receives a retico IU from AMQReader",
# #         ],
# #     ),
# #     # partial(
# #     #     cases,
# #     #     conditionss=[
# #     #         [
# #     #             ("module", ["AMQWriter Module"]),
# #     #             ("event", ["AMQWriter sends a message to ActiveMQ"]),
# #     #         ],
# #     #         [
# #     #             ("module", ["AMQReader Module"]),
# #     #             (
# #     #                 "event",
# #     #                 [
# #     #                     "AMQReader receives a message from ActiveMQ",
# #     #                     "AMQReader creates new iu",
# #     #                 ],
# #     #             ),
# #     #         ],
# #     #     ],
# #     # ),
# # ]


# def configurate_logger(log_path):
#     """
#     Configure structlog's logger and set general logging args (timestamps,
#     log level, etc.)

#     Args:
#         filename: (str): name of file to write to.

#         foldername: (str): name of folder to write to.
#     """

#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(message)s",
#         force=True,  # <-- HERE BECAUSE IMPORTING SOME LIBS LIKE COQUITTS WAS BLOCKING THE LOGGINGS SYSTEM
#     )

#     processors = (
#         [
#             structlog.processors.TimeStamper(fmt="iso"),
#             structlog.processors.add_log_level,
#             structlog.processors.dict_tracebacks,
#         ]
#         + LOG_FILTERS
#         + [structlog.dev.ConsoleRenderer(colors=True)]
#     )

#     structlog.configure(
#         processors=processors,
#         logger_factory=structlog.stdlib.LoggerFactory(),
#         wrapper_class=structlog.stdlib.BoundLogger,
#         cache_logger_on_first_use=True,
#     )

#     # Logger pour le terminal
#     terminal_logger = structlog.get_logger("terminal")
#     # self.terminal_logger = self.terminal_logger.bind(module=self.name())

#     # Configuration du logger pour le fichier
#     file_handler = logging.FileHandler(log_path, mode="a", encoding="UTF-8")
#     file_handler.setLevel(logging.INFO)

#     # Créer un logger standard sans handlers pour éviter la duplication des logs dans le terminal
#     file_only_logger = logging.getLogger("file_logger")
#     file_only_logger.addHandler(file_handler)
#     file_only_logger.propagate = (
#         False  # Empêche la propagation des logs vers les loggers parents
#     )

#     # Encapsuler ce logger avec structlog
#     file_logger = structlog.wrap_logger(
#         file_only_logger,
#         processors=[
#             structlog.processors.TimeStamper(fmt="iso"),
#             structlog.processors.add_log_level,
#             structlog.processors.dict_tracebacks,
#             structlog.processors.JSONRenderer(),  # Format JSON pour le fichier
#         ],
#     )
#     # self.file_logger = self.file_logger.bind(module=self.name())

#     return terminal_logger, file_logger


# t, f = configurate_logger("logs/test")


# from pathlib import Path
# import structlog
# import logging


# # LOG LEVEL AND FILTER
# structlog.configure(
#     wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
# )
# logger = structlog.get_logger()


# # can use a basic logger that has color and library like Rich and better-exception
# log = structlog.get_logger()
# log.info("hello, %s!", "world", key="value!", more_than_strings=[1, 2, 3])
# log.info("Processing data from the API")
# log.warning("Resource usage is nearing capacity")
# log.error("Failed to save the file. Please check permissions")
# log.critical("System has encountered a critical failure. Shutting down")

# structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(0))


# # BETTER EXCEPTION WITH RICH
# try:
#     d = {"x": 42}
#     print(d["y"])
# except Exception:
#     log.exception("poor me", stack_info=True)
# log.info("all better now!", stack_info=True)


# # can use a custom logger
# # class CustomPrintLogger:

# #     def msg(self, message):

# #         print(message)


# # def proc(logger, method_name, event_dict):

# #     print("I got called with", event_dict)

# #     return repr(event_dict)


# # log = structlog.wrap_logger(
# #     CustomPrintLogger(),
# #     wrapper_class=structlog.BoundLogger,
# #     processors=[proc],
# # )
# # log.msg("hello world")
# # log.bind(x=42)
# # log.msg("hello world")


# # CONFIGURATION OF RENDER LIKE COLORS ETC
# import colorama

# cr = structlog.dev.ConsoleRenderer(
#     columns=[
#         # Render the timestamp without the key name in yellow.
#         structlog.dev.Column(
#             "timestamp",
#             structlog.dev.KeyValueColumnFormatter(
#                 key_style=None,
#                 value_style=colorama.Fore.YELLOW,
#                 reset_style=colorama.Style.RESET_ALL,
#                 value_repr=str,
#             ),
#         ),
#         # Render the event without the key name in bright magenta.
#         structlog.dev.Column(
#             "event",
#             structlog.dev.KeyValueColumnFormatter(
#                 key_style=None,
#                 value_style=colorama.Style.BRIGHT + colorama.Fore.MAGENTA,
#                 reset_style=colorama.Style.RESET_ALL,
#                 value_repr=str,
#             ),
#         ),
#         # custom
#         structlog.dev.Column(
#             "more_than_strings",
#             structlog.dev.KeyValueColumnFormatter(
#                 key_style=colorama.Fore.GREEN,
#                 value_style=colorama.Fore.MAGENTA,
#                 reset_style=colorama.Style.RESET_ALL,
#                 value_repr=str,
#             ),
#         ),
#         # Default formatter for all keys not explicitly mentioned. The key is
#         # cyan, the value is green.
#         structlog.dev.Column(
#             "",
#             structlog.dev.KeyValueColumnFormatter(
#                 key_style=colorama.Fore.CYAN,
#                 value_style=colorama.Fore.GREEN,
#                 reset_style=colorama.Style.RESET_ALL,
#                 value_repr=str,
#             ),
#         ),
#     ]
# )

# structlog.configure(processors=structlog.get_config()["processors"][:-1] + [cr])
# log = structlog.get_logger()
# log.info("hello, %s!", "world", key="value!", more_than_strings=[1, 2, 3])
# log.info(f"hello, {13}!", key="value!", more_than_strings=[1, 2, 3])


# # JSON
# structlog.configure(
#     processors=[
#         structlog.processors.add_log_level,
#         structlog.processors.TimeStamper(fmt="iso"),
#         structlog.processors.JSONRenderer(),
#     ]
# )
# logger = structlog.get_logger()
# logger.info("Log entry in JSON format")


# # FILTER
# # def drop_messages(_, __, event_dict):
# #     if event_dict.get("route") == "login":
# #         raise structlog.DropEvent
# #     return event_dict


# # def drop_superior_messages(_, __, event_dict):
# #     if event_dict.get("id") <= 2:
# #         raise structlog.DropEvent
# #     return event_dict


# # structlog.configure(
# #     processors=[
# #         structlog.contextvars.merge_contextvars,
# #         structlog.processors.add_log_level,
# #         structlog.processors.TimeStamper(fmt="iso"),
# #         drop_messages,
# #         drop_superior_messages,
# #         structlog.dev.ConsoleRenderer(),
# #     ]
# # )
# # logger = structlog.get_logger()
# # logger.info("User created", name="John Doe", route="login", id=0)
# # logger.info("Post Created", title="My first post", route="post", id=1)
# # logger.info("Post Created", title="My first post", route="post", id=2)
# # logger.info("Post Created", title="My first post", route="post", id=3)
# # logger.info("Post Created", title="My first post", route="post", id=4)


# # # write logs in files

# # structlog.configure(
# #     processors=[
# #         structlog.contextvars.merge_contextvars,
# #         structlog.processors.add_log_level,
# #         structlog.processors.TimeStamper(fmt="iso"),
# #         drop_messages,
# #         drop_superior_messages,
# #         structlog.dev.ConsoleRenderer(),
# #     ],
# #     logger_factory=structlog.WriteLoggerFactory(file=Path("log_file.log").open("wt")),
# # )

# # logger.info("Post Created", title="My first post", route="post", id=4)


# ########  TEST 1 A SIMPLE THREAD
# ## Test threading and logs
# import threading
# import time


# def _run_thread():
#     l = structlog.get_logger("thread")
#     i = 0
#     while i < 1:
#         i += 1
#         t.info("test thread")
#         l.info("test thread")
#         time.sleep(2)
#     try:
#         0 / 0
#     except Exception:
#         t.exception("lala")
#         l.exception("lala")

#     0 / 0


# threading.Thread(target=_run_thread).start()


############
########  TEST 2 A SIMPLE RETICO MODULE
import retico_core
import threading
import time
import traceback


class ThreadTest(retico_core.abstract.AbstractProducingModule):

    @staticmethod
    def name():
        return "TestTextIUProducing Module"

    @staticmethod
    def description():
        return "A Module producing TextIU each n seconds"

    @staticmethod
    def output_iu():
        return retico_core.text.TextIU

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tts_thread_active = False
        self.cpt = 0

    def prepare_run(self):
        super().prepare_run()
        self._tts_thread_active = True
        threading.Thread(target=self.run_process).start()

    def shutdown(self):
        super().shutdown()
        self._tts_thread_active = False

    def process_update(self, update_message):
        pass

    def run_process(self):
        while self._tts_thread_active:
            try:

                # hardcoded IU values just for the test
                iu = self.create_iu(text=f"this is a test message : {self.cpt}")

                um = retico_core.UpdateMessage()
                um.add_iu(iu, retico_core.UpdateType.ADD)
                self.append(um)
                self.terminal_logger.info(
                    "thread exception",
                    text=iu.text,
                )
                self.cpt += 1

                try:
                    if self.cpt % 2 == 0:
                        raise TypeError("lala")
                except Exception as e:
                    log_exception(module=self, exception=e)

                if self.cpt % 2 == 0:
                    raise TypeError("lalalala")
                time.sleep(2)
            except Exception as e:
                log_exception(module=self, exception=e)


def test_structlog_1():
    testthread = ThreadTest()
    logger = structlog.get_logger()

    retico_core.network.LOG_FILTERS = [
        partial(
            filter_cases,
            cases=[
                [
                    ("level", ["error", "exception"]),
                ],
                [
                    (
                        "event",
                        [
                            "thread exception",
                            "thread exception 2",
                            "error",
                            "exception",
                        ],
                    ),
                ],
                [
                    ("exc_info", [True]),
                ],
            ],
        ),
    ]

    # running system
    try:
        retico_core.network.run(testthread)
        logger.info("Dialog system ready")
        # wait("q")
        input()
        retico_core.network.stop(testthread)
    except Exception:
        logger.exception("thread exception 2")
        testthread.terminal_logger.exception("thread exception 2")
        retico_core.network.stop(testthread)


test_structlog_1()

from keyboard import wait
import matplotlib
from retico_core import *
from retico_core.abstract import *
from retico_core.text import *
from retico_core.log_utils import *
import structlog
import torch

from amq_2 import TextAnswertoBEATBridge
from vad_turn import VADTurnModule
from whisper_asr_interruption import WhisperASRInterruptionModule


def test_structlog_2():
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

    network.LOG_FILTERS = [
        # partial(
        #     filter_value_not_in_list,
        #     key="event",
        #     values=[
        #         "error",
        #         "exception",
        #     ],
        # ),
        partial(
            filter_cases,
            cases=[
                [
                    ("level", ["error", "exception"]),
                ],
                [
                    (
                        "event",
                        ["thread exception", "error", "exception"],
                    ),
                ],
                [
                    ("exc_info", [True]),
                ],
            ],
        ),
    ]

    # running system
    try:
        network.run(mic, log_folder)
        logger.info("Dialog system ready")
        # wait("q")
        input()
        network.stop(mic)
    except Exception:
        logger.exception("test")
        mic.terminal_logger.exception("test")
        network.stop(mic)


# test_structlog_2()


# """
# C:\Users\Alafate\miniconda3\envs\retico_cuda\Lib\site-packages\structlog\_base.py:165: UserWarning: Remove `format_exc_info` from your processor chain if you want pretty exceptions.
#   event_dict = proc(self._logger, method_name, event_dict)
# Exception in thread Thread-5 (_run):
# Traceback (most recent call last):
#   File "C:\Users\Alafate\Documents\retico_test\amq_2.py", line 388, in process_update
#     self._previous_iu = output_iu
#                         ^^^^^^^^^
# UnboundLocalError: cannot access local variable 'output_iu' where it is not associated with a value
# """
