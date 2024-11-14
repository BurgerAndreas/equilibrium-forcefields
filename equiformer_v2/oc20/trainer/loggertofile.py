import logging



def close_all_log_handlers():
    # Get the root logger
    root_logger = logging.getLogger()

    # Remove handlers from the root logger
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Optionally, disable propagation for all loggers (to stop child loggers from propagating messages back to root)
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.propagate = False  # Disable propagation of log messages to parent loggers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

class FileLogger:
    def __init__(
        self, is_master=False, is_rank0=False, 
        output_dir=None, logger_name="training",
        log_to_file=True
    ):
        # only call by master
        # checked outside the class
        self.output_dir = output_dir
        if is_rank0:
            self.logger_name = logger_name
            self.logger = self.get_logger(
                output_dir, 
                log_to_file=is_master and log_to_file
            )
        else:
            self.logger_name = None
            self.logger = NoOp()

    def get_logger(self, output_dir, log_to_file):
        close_all_log_handlers()
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(logging.DEBUG)

        # for logging to file
        if output_dir and log_to_file:
            time_formatter = logging.Formatter(
                "%(asctime)s - %(filename)s:%(lineno)d - %(message)s",
                datefmt="%H:%M",
            )
            debuglog = logging.FileHandler(output_dir + "/debug.log")
            debuglog.setLevel(logging.DEBUG)
            debuglog.setFormatter(time_formatter)
            logger.addHandler(debuglog)

        # for logging to console
        formatter = logging.Formatter(
            "%(message)s",
            # datefmt="%Y-%m-%d %H:%M:%S",
            datefmt="%H:%M",
        )
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.DEBUG)
        logger.addHandler(console)

        # Reference: https://stackoverflow.com/questions/21127360/python-2-7-log-displayed-twice-when-logging-module-is-used-in-two-python-scri
        logger.propagate = False
        
        print(f"FileLogger handlers after setup:\n {logger.handlers}")

        return logger

    def console(self, *args):
        self.logger.debug(*args)

    def event(self, *args):
        self.logger.warn(*args)

    def verbose(self, *args):
        self.logger.info(*args)

    def info(self, *args):
        self.logger.info(*args)
    
    def warn(self, *args):
        self.logger.warn(*args)
    
    def warning(self, *args):
        self.logger.warning(*args)
    
    def error(self, *args):
        self.logger.error(*args)
    
    def close(self):
        # remove all handlers
        for hdlr in self.logger.handlers:
            self.logger.removeHandler(hdlr) 
            hdlr.close()
        print(f"FileLogger hasHandlers after close: {self.logger.handlers}")
        del self.logger
        close_all_log_handlers()



# no_op method/object that accept every signature
class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs):
            pass

        return no_op
