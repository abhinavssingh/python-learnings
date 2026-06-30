from lib.utility.machinelearning._logging import ExceptionLoggingMixin


class BaseImbalanceHandler(ExceptionLoggingMixin):

    def get_pipeline_step(self):
        raise NotImplementedError("Must implement get_pipeline_step()")
