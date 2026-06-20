class BaseImbalanceHandler:

    def get_pipeline_step(self):
        raise NotImplementedError("Must implement get_pipeline_step()")
