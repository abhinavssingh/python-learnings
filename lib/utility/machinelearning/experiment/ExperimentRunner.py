class ExperimentRunner:

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.results = []
        self.tuned_results = []

    def run(self, model_name, wrapper, X_train, X_test, y_train, y_test):

        wrapper.build_pipeline(self.preprocessor)
        wrapper.train(X_train, y_train)

        preds = wrapper.predict(X_test)
        metrics = wrapper.evaluate(y_test, preds)

        result = {
            "model": model_name,
            **metrics
        }

        self.results.append(result)
        return result

    def run_all(self, model_registry, X_train, X_test, y_train, y_test):
        for name, wrapper in model_registry.items():
            self.run(name, wrapper, X_train, X_test, y_train, y_test)

        return self.results
