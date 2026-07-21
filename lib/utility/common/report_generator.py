class ReportGenerator:

    @staticmethod
    def generate(result):

        return {
            "model_name": result.model_name,
            "training_time": result.training_time,
            **result.metrics
        }
