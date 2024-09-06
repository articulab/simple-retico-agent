from pathlib import Path
import structlog
import logging


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
# logger_json = structlog.get_logger("JSON")
# logger_json.info("Log entry in JSON format")


# # FILTER
# def drop_messages(_, __, event_dict):
#     if event_dict.get("route") == "login":
#         raise structlog.DropEvent
#     return event_dict


# def drop_superior_messages(_, __, event_dict):
#     if event_dict.get("id") <= 2:
#         raise structlog.DropEvent
#     return event_dict


# structlog.configure(
#     processors=[
#         structlog.contextvars.merge_contextvars,
#         structlog.processors.add_log_level,
#         structlog.processors.TimeStamper(fmt="iso"),
#         drop_messages,
#         drop_superior_messages,
#         structlog.dev.ConsoleRenderer(),
#     ]
# )
# logger_console = structlog.get_logger("console")
# logger_console.info("User created", name="John Doe", route="login", id=0)
# logger_console.info("Post Created", title="My first post", route="post", id=1)
# logger_console.info("Post Created", title="My first post", route="post", id=2)
# logger_console.info("Post Created", title="My first post", route="post", id=3)
# logger_console.info("Post Created", title="My first post", route="post", id=4)


# # write logs in files

# structlog.configure(
#     processors=[
#         structlog.contextvars.merge_contextvars,
#         structlog.processors.add_log_level,
#         structlog.processors.TimeStamper(fmt="iso"),
#         drop_messages,
#         drop_superior_messages,
#         # structlog.dev.ConsoleRenderer(),
#         structlog.processors.JSONRenderer(),
#     ],
#     logger_factory=structlog.WriteLoggerFactory(file=Path("log_file.log").open("wt")),
# )
# logger_json = structlog.get_logger("json")
# logger_json.info("Post Created", title="My first post", route="post", id=4)
# logger_console = structlog.get_logger("console")
# logger_console.info("Post Created", title="My first post", route="post", id=4)

###

import logging.config
import structlog

# logging.config.dictConfig(
#     {
#         "version": 1,
#         "disable_existing_loggers": False,
#         "handlers": {
#             "default": {
#                 "level": "DEBUG",
#                 "class": "logging.StreamHandler",
#             },
#             "file": {
#                 "level": "DEBUG",
#                 "class": "logging.handlers.WatchedFileHandler",
#                 "filename": "test.log",
#             },
#         },
#         "loggers": {
#             "": {
#                 "handlers": ["default", "file"],
#                 "level": "DEBUG",
#                 "propagate": True,
#             },
#         },
#     }
# )
# structlog.configure(
#     processors=[
#         structlog.stdlib.add_log_level,
#         structlog.stdlib.PositionalArgumentsFormatter(),
#         structlog.processors.TimeStamper(fmt="iso"),
#         structlog.processors.StackInfoRenderer(),
#         structlog.processors.format_exc_info,
#         structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
#     ],
#     logger_factory=structlog.stdlib.LoggerFactory(),
#     wrapper_class=structlog.stdlib.BoundLogger,
#     cache_logger_on_first_use=True,
# )
# structlog.get_logger("test").info("hello")


###

# timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S")
# shared_processors = [
#     structlog.stdlib.add_log_level,
#     timestamper,
# ]

# structlog.configure(
#     processors=shared_processors
#     + [
#         structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
#     ],
#     logger_factory=structlog.stdlib.LoggerFactory(),
#     cache_logger_on_first_use=True,
# )

# formatter = structlog.stdlib.ProcessorFormatter(
#     # These run ONLY on `logging` entries that do NOT originate within
#     # structlog.
#     foreign_pre_chain=shared_processors,
#     # These run on ALL entries after the pre_chain is done.
#     processors=[
#         # Remove _record & _from_structlog.
#         structlog.stdlib.ProcessorFormatter.remove_processors_meta,
#         structlog.dev.ConsoleRenderer(),
#     ],
# )

# logging.getLogger("stdlog").info("woo")
# structlog.get_logger("structlog").info("amazing", events="oh yes")

#########

# import logging
# import structlog

# # Configuration du logging standard
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(message)s",
#     handlers=[
#         logging.StreamHandler(),  # Pour afficher les logs dans le terminal
#         logging.FileHandler("app.log"),  # Pour écrire les logs dans un fichier local
#     ],
# )

# # Configuration de structlog
# structlog.configure(
#     processors=[
#         structlog.processors.TimeStamper(fmt="iso"),
#         structlog.processors.JSONRenderer(),  # Pour formater les logs en JSON
#     ],
#     logger_factory=structlog.stdlib.LoggerFactory(),
#     wrapper_class=structlog.stdlib.BoundLogger,
#     cache_logger_on_first_use=True,
# )

# # Utilisation de structlog
# logger = structlog.get_logger()

# # Exemple de log
# logger.info("This is an info log.", module="user_login", user="john_doe")


####
# import logging
# import structlog

# # Configuration du logging standard
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(message)s",
# )

# # Configuration du logger pour le terminal
# structlog.configure(
#     processors=[
#         structlog.processors.TimeStamper(fmt="iso"),
#         structlog.dev.ConsoleRenderer(
#             colors=True
#         ),  # Format lisible et coloré pour le terminal
#     ],
#     logger_factory=structlog.stdlib.LoggerFactory(),
#     wrapper_class=structlog.stdlib.BoundLogger,
#     cache_logger_on_first_use=True,
# )

# # Logger pour le terminal
# terminal_logger = structlog.get_logger("terminal")

# # Configuration du logger pour le fichier
# file_logger = structlog.get_logger("file")
# file_handler = logging.FileHandler("app.log")
# file_handler.setFormatter(logging.Formatter("%(message)s"))

# file_logger = structlog.wrap_logger(
#     file_handler,
#     processors=[
#         structlog.processors.TimeStamper(fmt="iso"),
#         structlog.processors.JSONRenderer(),  # Format JSON pour le fichier
#     ],
#     logger_factory=structlog.stdlib.LoggerFactory(),
# )

# # Utilisation des loggers
# terminal_logger.info("This is a terminal log.", module="user_login", user="john_doe")
# file_logger.info("This is a file log.", module="user_login", user="john_doe")


###

# import logging
# import structlog

# # Configuration du logging standard pour le terminal
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(message)s",
# )

# # Configuration du logger pour le terminal
# structlog.configure(
#     processors=[
#         structlog.processors.TimeStamper(fmt="iso"),
#         structlog.dev.ConsoleRenderer(
#             colors=True
#         ),  # Format lisible et coloré pour le terminal
#     ],
#     logger_factory=structlog.stdlib.LoggerFactory(),
#     wrapper_class=structlog.stdlib.BoundLogger,
#     cache_logger_on_first_use=True,
# )

# # Logger pour le terminal
# terminal_logger = structlog.get_logger("terminal")

# # Configuration du logger pour le fichier
# file_handler = logging.FileHandler("app.log")
# file_handler.setLevel(logging.INFO)

# file_logger = structlog.wrap_logger(
#     logging.getLogger("file_logger"),
#     processors=[
#         structlog.processors.TimeStamper(fmt="iso"),
#         structlog.processors.JSONRenderer(),  # Format JSON pour le fichier
#     ],
# )

# file_logger._logger.addHandler(file_handler)

# # Utilisation des loggers
# terminal_logger.info("This is a terminal log.", module="user_login", user="john_doe")
# file_logger.info("This is a file log.", module="user_login", user="john_doe")


###

# import logging
# import structlog

# # Configuration du logging standard pour le terminal
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(message)s",
# )

# # Configuration du logger pour le terminal
# structlog.configure(
#     processors=[
#         structlog.processors.TimeStamper(fmt="iso"),
#         structlog.dev.ConsoleRenderer(
#             colors=True
#         ),  # Format lisible et coloré pour le terminal
#     ],
#     logger_factory=structlog.stdlib.LoggerFactory(),
#     wrapper_class=structlog.stdlib.BoundLogger,
#     cache_logger_on_first_use=True,
# )

# # Logger pour le terminal
# terminal_logger = structlog.get_logger("terminal")

# # Configuration du logger pour le fichier
# file_handler = logging.FileHandler("app.log")
# file_handler.setLevel(logging.INFO)

# # Créer un logger standard sans handlers pour éviter la duplication des logs dans le terminal
# file_only_logger = logging.getLogger("file_logger")
# file_only_logger.setLevel(logging.INFO)
# file_only_logger.addHandler(file_handler)
# file_only_logger.propagate = (
#     False  # Empêche la propagation des logs vers les loggers parents
# )

# # Encapsuler ce logger avec structlog
# file_logger = structlog.wrap_logger(
#     file_only_logger,
#     processors=[
#         structlog.processors.TimeStamper(fmt="iso"),
#         structlog.processors.JSONRenderer(),  # Format JSON pour le fichier
#     ],
# )

# # Utilisation des loggers
# terminal_logger.info("This is a terminal log.", module="user_login", user="john_doe")
# file_logger.info("This is a file log.", module="user_login", user="john_doe")


### tentative de simplification

import logging
import structlog

# Configuration du logging standard pour le terminal
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
)


def drop_messages(_, __, event_dict):
    if event_dict.get("route") == "login":
        raise structlog.DropEvent
    return event_dict


def drop_superior_messages(_, __, event_dict):
    if event_dict.get("id"):
        if event_dict.get("id") <= 2:
            raise structlog.DropEvent
    return event_dict


# structlog.configure(
#     processors=[
#         structlog.contextvars.merge_contextvars,
#         structlog.processors.add_log_level,
#         structlog.processors.TimeStamper(fmt="iso"),
#         drop_messages,
#         drop_superior_messages,
#         structlog.dev.ConsoleRenderer(),
#     ],
#     logger_factory=structlog.stdlib.LoggerFactory(),
#     wrapper_class=structlog.stdlib.BoundLogger,
#     cache_logger_on_first_use=True,
# )

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        drop_messages,
        drop_superior_messages,
        structlog.dev.ConsoleRenderer(
            colors=True
        ),  # Format lisible et coloré pour le terminal
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Logger pour le terminal
terminal_logger = structlog.get_logger("terminal")

# Configuration du logger pour le fichier
file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.DEBUG)

# Créer un logger standard sans handlers pour éviter la duplication des logs dans le terminal
file_only_logger = logging.getLogger("file_logger")
file_only_logger.addHandler(file_handler)
file_only_logger.propagate = (
    False  # Empêche la propagation des logs vers les loggers parents
)

# Encapsuler ce logger avec structlog
file_logger = structlog.wrap_logger(
    file_only_logger,
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.dict_tracebacks,
        structlog.processors.JSONRenderer(),  # Format JSON pour le fichier
    ],
    # logger_factory=structlog.stdlib.LoggerFactory(),
)

# Utilisation des loggers
terminal_logger.info("This is a terminal log.", module="user_login", user="john_doe")
file_logger.info("This is a file log.", module="user_login", user="john_doe")

# try:
#     1234 / 0
# except Exception:
#     terminal_logger.exception("test")
#     file_logger.exception("test")


class A:

    def __init__(self):
        self.name = "lala"
        self.mode = "2"


a = A()
print(a.__dict__)

terminal_logger.info(
    "This is a terminal log.", module="user_login", user="john_doe", **a.__dict__
)
terminal_logger = structlog.get_logger("console")
terminal_logger.info("User created", name="John Doe", route="login", id=0)
terminal_logger.info("Post Created", title="My first post", route="post", id=1)
terminal_logger.info("Post Created", title="My first post", route="post", id=2)
terminal_logger.info("Post Created", title="My first post", route="post", id=3)
terminal_logger.info("Post Created", title="My first post", route="post", id=4)
terminal_logger.warning("Post Created", title="My first post", route="post", id=4)
terminal_logger.error("Post Created", title="My first post", route="post", id=4)
