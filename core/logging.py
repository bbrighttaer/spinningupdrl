import logging
import logging.handlers

import colorlog


class Logger:
    def __init__(self, trial_name, working_dir, level) -> None:
        self.root = logging.getLogger(trial_name)
        if not self.root.hasHandlers():
            # create handlers
            self.file_handler = logging.handlers.RotatingFileHandler(filename=f"{working_dir}/{trial_name}.log",
                                                                     mode='a')
            self.console_handler = logging.StreamHandler()

            # create formatter
            formatter = colorlog.ColoredFormatter(
                # f'%(log_color)s [%(asctime)s] %(levelname)-3s %(name)s: %(message)s',
                # f'%(log_color)s [%(asctime)s]: %(message)s',
                fmt="%(log_color)s[%(asctime)s]\n%(message)s",
                datefmt='%y-%m-%d %H:%M:%s',
                log_colors={
                    "DEBUG": "white",
                    "INFO": "light_white",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                }
            )

            # associate handlers with formatter
            self.file_handler.setFormatter(formatter)
            self.console_handler.setFormatter(formatter)
            self.file_handler.setLevel(logging.INFO)

            # add handlers to root logger
            self.root.addHandler(self.file_handler)
            self.root.addHandler(self.console_handler)

            # root logger level
            self.root.setLevel(level)

    def info(self, msg):
        self.root.info(msg)

    def debug(self, msg):
        self.root.debug(msg)

    def error(self, msg):
        self.root.error(msg)

    def warning(self, msg):
        self.root.warning(msg)

    def warn(self, msg):
        self.root.warning(msg)
