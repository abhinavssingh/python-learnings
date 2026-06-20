class ResultBuilder:

    @staticmethod
    def build(
        *,
        model,
        family="unknown",
        experiment=None,
        mode="train-test",
        result_type="baseline",
        imbalance_summary=None,
        artifacts=None,
        extra=None,
        **metrics
    ):
        """
        Generic result builder for all experiment types.
        """

        # ✅ Default experiment naming
        if not experiment:
            experiment = f"{model} | classification"

        result = {
            "model": model,
            "family": family,
            "experiment": experiment,
            "mode": mode,
            "type": result_type,

            # ✅ Imbalance flags
            "imbalance_applied": imbalance_summary is not None,
            "imbalance_method": (
                imbalance_summary.get("method")
                if imbalance_summary else None
            ),

            # ✅ Metrics (dynamic)
            **metrics,

            # ✅ Artifacts (always present)
            "artifacts": artifacts or {}
        }

        # ✅ Attach imbalance details inside artifacts
        if imbalance_summary:
            result["artifacts"]["imbalance"] = imbalance_summary

        # ✅ Optional extra metadata (ensemble, tuning, etc.)
        if extra:
            result.update(extra)

        return result
