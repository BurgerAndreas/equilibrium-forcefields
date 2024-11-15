import logging


class FileLogger:
    """Wrapper around logging module to log to console and file.
    Access via
    _log = FileLogger(is_master, is_rank0, output_dir, logger_name)
    _log.console("message")
    _log.info("message2")
    _log.logger.handlers[0].flush()
    """

    def __init__(
        self, is_master=False, is_rank0=False, output_dir=None, logger_name="training"
    ):
        # only call by master
        # checked outside the class
        self.output_dir = output_dir
        if is_rank0:
            self.logger_name = logger_name
            self.logger = self.get_logger(output_dir, log_to_file=is_master)
        else:
            self.logger_name = None
            self.logger = NoOp()

    def get_logger(self, output_dir, log_to_file):
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(message)s", datefmt="%H:%M")

        if output_dir and log_to_file:

            time_formatter = logging.Formatter(
                "%(asctime)s - %(filename)s:%(lineno)d - %(message)s",
                datefmt="%H:%M",
            )
            debuglog = logging.FileHandler(output_dir + "/debug.log")
            debuglog.setLevel(logging.DEBUG)
            debuglog.setFormatter(time_formatter)
            logger.addHandler(debuglog)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.DEBUG)
        logger.addHandler(console)

        # Reference: https://stackoverflow.com/questions/21127360/python-2-7-log-displayed-twice-when-logging-module-is-used-in-two-python-scri
        logger.propagate = False
        
        print(f"FileLogger: {logger}")
        print(f"FileLogger handlers: {logger.handlers}")

        return logger

    def console(self, *args):
        self.logger.debug(*args)

    def event(self, *args):
        self.logger.warn(*args)

    def verbose(self, *args):
        self.logger.info(*args)

    def info(self, *args):
        self.logger.info(*args)
    
    def warning(self, *args):
        self.logger.warning(*args)
        
    def warn(self, *args):
        self.logger.warn(*args)
    
    def error(self, *args):
        self.logger.error(*args)
    
    def close(self):
        # remove all handlers
        for hdlr in self.logger.handlers:
            self.logger.removeHandler(hdlr) 
            hdlr.close()
        print(f"FileLogger hasHandlers: {self.logger.hasHandlers()}")


# no_op method/object that accept every signature
class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs):
            pass

        return no_op