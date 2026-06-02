import pandas as pd


class ModelComparator:

    def __init__(self, results):
        self.df = pd.DataFrame(results)

    def rank(self, metric="R2", ascending=False):
        return self.df.sort_values(metric, ascending=ascending).reset_index(drop=True)

    def best_model(self, metric="R2"):
        ranked = self.rank(metric, ascending=(metric == "MSE"))
        return ranked.iloc[0].to_dict()

    def compare(self):
        return self.df.groupby("model").mean(numeric_only=True)

    def baseline_vs_tuned(self):
        baseline = self.df[self.df["type"] == "baseline"]
        tuned = self.df[self.df["type"] == "tuned"]

        results = []

        for model in self.df["model"].unique():

            b = baseline[baseline["model"] == model]
            t = tuned[tuned["model"] == model]

            if b.empty or t.empty:
                continue

            b_best = b.sort_values("R2", ascending=False).iloc[0]
            t_best = t.sort_values("R2", ascending=False).iloc[0]

            results.append({
                "model": model,
                "baseline_R2": b_best["R2"],
                "tuned_R2": t_best["R2"],
                "delta_R2": t_best["R2"] - b_best["R2"]
            })

        return pd.DataFrame(results)
