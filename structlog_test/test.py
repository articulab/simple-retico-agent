from pathlib import Path
import structlog
import logging


# LOG LEVEL AND FILTER
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
)
logger = structlog.get_logger()


# can use a basic logger that has color and library like Rich and better-exception
log = structlog.get_logger()
log.info("hello, %s!", "world", key="value!", more_than_strings=[1, 2, 3])
log.info("Processing data from the API")
log.warning("Resource usage is nearing capacity")
log.error("Failed to save the file. Please check permissions")
log.critical("System has encountered a critical failure. Shutting down")

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(0))


# BETTER EXCEPTION WITH RICH
try:
    d = {"x": 42}
    print(d["y"])
except Exception:
    log.exception("poor me", stack_info=True)
log.info("all better now!", stack_info=True)


# can use a custom logger
# class CustomPrintLogger:

#     def msg(self, message):

#         print(message)


# def proc(logger, method_name, event_dict):

#     print("I got called with", event_dict)

#     return repr(event_dict)


# log = structlog.wrap_logger(
#     CustomPrintLogger(),
#     wrapper_class=structlog.BoundLogger,
#     processors=[proc],
# )
# log.msg("hello world")
# log.bind(x=42)
# log.msg("hello world")


# CONFIGURATION OF RENDER LIKE COLORS ETC
import colorama

cr = structlog.dev.ConsoleRenderer(
    columns=[
        # Render the timestamp without the key name in yellow.
        structlog.dev.Column(
            "timestamp",
            structlog.dev.KeyValueColumnFormatter(
                key_style=None,
                value_style=colorama.Fore.YELLOW,
                reset_style=colorama.Style.RESET_ALL,
                value_repr=str,
            ),
        ),
        # Render the event without the key name in bright magenta.
        structlog.dev.Column(
            "event",
            structlog.dev.KeyValueColumnFormatter(
                key_style=None,
                value_style=colorama.Style.BRIGHT + colorama.Fore.MAGENTA,
                reset_style=colorama.Style.RESET_ALL,
                value_repr=str,
            ),
        ),
        # custom
        structlog.dev.Column(
            "more_than_strings",
            structlog.dev.KeyValueColumnFormatter(
                key_style=colorama.Fore.GREEN,
                value_style=colorama.Fore.MAGENTA,
                reset_style=colorama.Style.RESET_ALL,
                value_repr=str,
            ),
        ),
        # Default formatter for all keys not explicitly mentioned. The key is
        # cyan, the value is green.
        structlog.dev.Column(
            "",
            structlog.dev.KeyValueColumnFormatter(
                key_style=colorama.Fore.CYAN,
                value_style=colorama.Fore.GREEN,
                reset_style=colorama.Style.RESET_ALL,
                value_repr=str,
            ),
        ),
    ]
)

structlog.configure(processors=structlog.get_config()["processors"][:-1] + [cr])
log = structlog.get_logger()
log.info("hello, %s!", "world", key="value!", more_than_strings=[1, 2, 3])
log.info(f"hello, {13}!", key="value!", more_than_strings=[1, 2, 3])


# JSON
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()
logger.info("Log entry in JSON format")


# FILTER
def drop_messages(_, __, event_dict):
    if event_dict.get("route") == "login":
        raise structlog.DropEvent
    return event_dict


def drop_superior_messages(_, __, event_dict):
    if event_dict.get("id") <= 2:
        raise structlog.DropEvent
    return event_dict


structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        drop_messages,
        drop_superior_messages,
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger()
logger.info("User created", name="John Doe", route="login", id=0)
logger.info("Post Created", title="My first post", route="post", id=1)
logger.info("Post Created", title="My first post", route="post", id=2)
logger.info("Post Created", title="My first post", route="post", id=3)
logger.info("Post Created", title="My first post", route="post", id=4)


# write logs in files

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        drop_messages,
        drop_superior_messages,
        structlog.dev.ConsoleRenderer(),
    ],
    logger_factory=structlog.WriteLoggerFactory(file=Path("log_file.log").open("wt")),
)

logger.info("Post Created", title="My first post", route="post", id=4)
