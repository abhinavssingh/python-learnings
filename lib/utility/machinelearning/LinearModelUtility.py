import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lib.utility.machinelearning.HyperparameterTuner import HyperparameterTuner


class LinearModelUtility:
    """
    ✅ Refactored ML Utility (Reusable + Experiment-Driven)

    Features:
    - One-time data preparation
    - Stateless pipeline creation
    - Train-test + K-Fold
    - Run multiple experiments without re-instantiation
    - Easy comparison output (DataFrame-ready)
    """

    # ---------------------------------------------------
    # INIT
    # ---------------------------------------------------
    def __init__(self, df, target_col, imputer=None, outlier_handler=None):
        self.df = df
        self.target_col = target_col
        self.experiment_results = []

        # ✅ DEFAULT GLOBAL TRANSFORMERS
        self.default_imputer = imputer
        self.default_outlier = outlier_handler

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.num_cols = None
        self.cat_cols = None

        # Model registry
        self.model_registry = {
            "LinearRegression": LinearRegression(),
            "SGDRegressor": SGDRegressor(max_iter=1000, tol=1e-3),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
        }

    # ---------------------------------------------------
    # STEP 1: PREPARE DATA (CALL ONCE)
    # ---------------------------------------------------
    def prepare_data(self, test_size=0.2, random_state=42):
        X = self.df.drop(self.target_col, axis=1)
        y = self.df[self.target_col]

        self.num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    # ---------------------------------------------------
    # STEP 2: PREPROCESSOR FACTORY (STATELESS ✅)
    # ---------------------------------------------------
    def create_preprocessor(self, imputer=None, outlier_handler=None):

        steps = []

        if imputer is not None:
            steps.append(('imputer', imputer))

        if outlier_handler is not None:
            steps.append(('outlier', outlier_handler))

        numeric_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        column_transform = ColumnTransformer([
            ('num', numeric_pipeline, self.num_cols),
            ('cat', categorical_pipeline, self.cat_cols)
        ])

        steps.append(('column_transform', column_transform))

        return Pipeline(steps)

    # ---------------------------------------------------
    # PIPELINE BUILDER
    # ---------------------------------------------------
    def build_pipeline(self, model_name, imputer=None, outlier_handler=None):

        if model_name not in self.model_registry:
            raise ValueError(f"❌ Model '{model_name}' not found")

        # ✅ fallback to defaults
        if imputer is None:
            imputer = self.default_imputer

        if outlier_handler is None:
            outlier_handler = self.default_outlier

        preprocessor = self.create_preprocessor(imputer, outlier_handler)

        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('model', self.model_registry[model_name])
        ])

        return pipeline

    # ---------------------------------------------------
    # RUN SINGLE EXPERIMENT ✅
    # ---------------------------------------------------
    def run_experiment(self, model_name, imputer=None, outlier_handler=None, k_fold=None):

        pipeline = self.build_pipeline(model_name, imputer, outlier_handler)

        # ✅ K-FOLD MODE
        if k_fold:
            kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)

            # ✅ Out-of-fold predictions (only ONE training cycle)
            y_pred = cross_val_predict(
                pipeline,
                self.X_train,
                self.y_train,
                cv=kf,
                n_jobs=8
            )

            y_true = self.y_train

            # ✅ Compute metrics directly (faster ✅)
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)

            result = {
                "model": model_name,
                "mode": "k-fold",
                "type": "baseline",
                "k": k_fold,
                "R2": r2,
                "MSE": mse,
            }

            # ✅ store globally
            self.experiment_results.append(result)
            return result

        # ✅ TRAIN-TEST MODE
        pipeline.fit(self.X_train, self.y_train)
        y_true = self.y_train
        y_pred = pipeline.predict(self.X_test)

        result = {
            "model": model_name,
            "mode": "train-test",
            "type": "baseline",
            "R2": r2_score(self.y_test, y_pred),
            "MSE": mean_squared_error(self.y_test, y_pred),
        }
        self.experiment_results.append(result)
        return result

    # ---------------------------------------------------
    # RUN MULTIPLE EXPERIMENTS ✅ (KEY FEATURE)
    # ---------------------------------------------------
    def run_experiments(self, configs):

        results = []

        for config in configs:
            result = self.run_experiment(**config)

            # Attach config info
            result.update({
                "imputer": type(config.get("imputer")).__name__ if config.get("imputer") else None,
                "outlier_handler": type(config.get("outlier_handler")).__name__ if config.get("outlier_handler") else None
            })

            results.append(result)

        return pd.DataFrame(results)

    # ---------------------------------------------------
    # TRAIN ALL MODELS (QUICK COMPARE)
    # ---------------------------------------------------
    def run_all_models(self, k_fold=None):

        results = []

        for model_name in self.model_registry.keys():
            result = self.run_experiment(model_name, k_fold=k_fold)
            results.append(result)

        return pd.DataFrame(results)

    # ---------------------------------------------------
    # MODEL DESCRIPTION
    # ---------------------------------------------------
    def get_description(self, model_name):
        descriptions = {
            "LinearRegression": "OLS: baseline model, no regularization",
            "SGDRegressor": "Efficient for large-scale data",
            "Ridge": "L2 regularization, reduces overfitting",
            "Lasso": "L1 regularization, performs feature selection",
            "ElasticNet": "Combination of L1 + L2"
        }
        return descriptions.get(model_name, "No description available")

    # ---------------------------------------------------
    # GRID SEARCH (DIRECT)
    # ---------------------------------------------------
    def grid_search_cv(self, model_name, param_grid, imputer=None, outlier_handler=None, cv=5, scoring='r2'):
        """
        Direct GridSearchCV (simpler alternative to HyperparameterTuner)

        Returns:
        - best params
        - best score
        - best model
        - test evaluation
        """

        # ✅ Build pipeline with defaults
        pipeline = self.build_pipeline(model_name, imputer=imputer, outlier_handler=outlier_handler)

        # ✅ Grid Search
        grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=8)

        grid.fit(self.X_train, self.y_train)

        result = {
            "mode": "gridsearch_simple",
            "model": model_name,
            "type": "tuned",
            "best_params": grid.best_params_,
            "best_score_cv": grid.best_score_,
            "cv_results": grid.cv_results_
        }

        # ✅ Evaluate on test set
        if self.X_test is not None and self.y_test is not None:
            y_pred = grid.best_estimator_.predict(self.X_test)

            result["test_metrics"] = {
                "R2": r2_score(self.y_test, y_pred),
                "MSE": mean_squared_error(self.y_test, y_pred)
            }
        self.experiment_results.append(result)
        return result

    # ---------------------------------------------------
    # HYPERPARAMETER TUNING (GRID & RANDOM)
    # ---------------------------------------------------

    def tune_model(self, model_name, param_grid, search_type="grid", imputer=None, outlier_handler=None, cv=5, n_iter=20, scoring="r2"):
        """
        Tune hyperparameters using GridSearchCV or RandomizedSearchCV.

        Parameters:
        - model_name: str
        - param_grid: dict
        - search_type: "grid" or "random"
        - imputer, outlier_handler: optional overrides
        - cv: folds
        - n_iter: for randomized search
        - scoring: scoring metric

        Returns:
        dict with best params, scores, and model
        """

        # ✅ build pipeline with defaults
        pipeline = self.build_pipeline(model_name, imputer=imputer, outlier_handler=outlier_handler)

        # ✅ create tuner instance
        tuner = HyperparameterTuner(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test
        )

        # ✅ choose search type
        if search_type == "grid":
            result = tuner.grid_search(pipeline=pipeline, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=8)
            self.experiment_results.append(result)

        elif search_type == "random":
            result = tuner.random_search(pipeline=pipeline, param_distributions=param_grid, cv=cv, n_iter=n_iter, scoring=scoring, n_jobs=8)
            self.experiment_results.append(result)

        else:
            raise ValueError("search_type must be 'grid' or 'random'")

        # ✅ attach metadata
        result.update({
            "model": model_name,
            "type": "tuned",
            "search_type": search_type
        })

        return result

    # ---------------------------------------------------
    # MODEL COMPARISON & RANKING
    # ---------------------------------------------------
    def rank_models(self, metric="R2", ascending=False):
        """
        Rank models based on a metric.

        metric:
            - "R2" → higher is better
            - "MSE" → lower is better
        """

        if not self.experiment_results:
            print("No experiment results found.")
            return None

        df = pd.DataFrame(self.get_all_flat_results())

        if metric not in df.columns:
            raise ValueError(f"{metric} not found in results")

        ranked = df.sort_values(metric, ascending=ascending).reset_index(drop=True)

        return ranked

    # ---------------------------------------------------
    # GET BEST MODEL
    # ---------------------------------------------------

    def get_best_model(self, metric="R2"):
        """
        Get best model based on metric
        """

        ranked = self.rank_models(metric=metric, ascending=(metric == "MSE"))

        if ranked is None or ranked.empty:
            return None

        return ranked.iloc[0].to_dict()

    def flatten_result(self, result):
        flat = {}

        for k, v in result.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    flat[sk] = sv
            else:
                flat[k] = v

        return flat

    def get_all_flat_results(self):
        return [self.flatten_result(r) for r in self.experiment_results]

    # ---------------------------------------------------
    # MODEL COMPARISON (AGGREGATE + BASELINE VS TUNED)
    # ---------------------------------------------------
    def compare_models(self):
        """
        Aggregate comparison per model
        """

        if not self.experiment_results:
            return None

        df = pd.DataFrame(self.get_all_flat_results())

        comparison = df.groupby("model").agg({
            "R2": ["mean", "max"],
            "MSE": ["mean", "min"]
        }).sort_values(by=("R2", "max"), ascending=False)

        return comparison

    def get_results_dict(self):
        """
        Structured dictionary for reporting
        """

        return {
            "all_results": self.get_all_flat_results(),
            "ranking_R2": self.rank_models("R2").to_dict(),
            "ranking_MSE": self.rank_models("MSE", ascending=True).to_dict(),
            "best_model_R2": self.get_best_model("R2"),
            "best_model_MSE": self.get_best_model("MSE")
        }

    # ---------------------------------------------------
        # COMPARISON: BASELINE VS TUNED
    # ---------------------------------------------------
    def get_combined_results_df(self):
        """
        Return a single consolidated DataFrame with:
        - all results
        - rankings
        - best model flags
        """

        df = pd.DataFrame(self.get_all_flat_results())

        if df.empty:
            return df

        # ✅ Ranking
        df["rank_R2"] = df["R2"].rank(ascending=False, method="dense")
        df["rank_MSE"] = df["MSE"].rank(ascending=True, method="dense")

        # ✅ Best flags
        best_r2_idx = df["R2"].idxmax()
        best_mse_idx = df["MSE"].idxmin()

        df["is_best_R2"] = False
        df["is_best_MSE"] = False

        df.loc[best_r2_idx, "is_best_R2"] = True
        df.loc[best_mse_idx, "is_best_MSE"] = True

        # ✅ Optional: improvement tracking (if exists)
        if "type" in df.columns:
            df["is_tuned"] = df["type"] == "tuned"

        # ✅ Sort (default: best R2 first)
        df = df.sort_values(by="R2", ascending=False).reset_index(drop=True)

        return df

    # ---------------------------------------------------
    # BASELINE VS TUNED COMPARISON
    # ---------------------------------------------------
    def compare_baseline_vs_tuned(self):
        """
        Compare baseline vs tuned models
        """

        if not self.experiment_results:
            print("No results found.")
            return None

        df = pd.DataFrame(self.get_all_flat_results())

        # ✅ Separate baseline & tuned
        baseline_df = df[df["type"] == "baseline"]
        tuned_df = df[df["type"] == "tuned"]

        comparisons = []

        for model in df["model"].unique():

            base = baseline_df[baseline_df["model"] == model]
            tuned = tuned_df[tuned_df["model"] == model]

            if base.empty or tuned.empty:
                continue

            # Take best baseline & tuned
            base_best = base.sort_values("R2", ascending=False).iloc[0]
            tuned_best = tuned.sort_values("R2", ascending=False).iloc[0]

            delta_r2 = tuned_best["R2"] - base_best["R2"]
            delta_mse = base_best["MSE"] - tuned_best["MSE"]

            comparisons.append({
                "model": model,

                # baseline
                "baseline_R2": base_best["R2"],
                "baseline_MSE": base_best["MSE"],

                # tuned
                "tuned_R2": tuned_best["R2"],
                "tuned_MSE": tuned_best["MSE"],

                # improvements
                "delta_R2": delta_r2,
                "delta_MSE": delta_mse,

                # percentage improvement
                "%_R2_improvement": (delta_r2 / abs(base_best["R2"])) * 100 if base_best["R2"] != 0 else None,
                "%_MSE_reduction": (delta_mse / base_best["MSE"]) * 100 if base_best["MSE"] != 0 else None
            })

        return pd.DataFrame(comparisons)

    # ---------------------------------------------------
    # BEST IMPROVEMENT MODEL
    # ---------------------------------------------------
    def best_improvement_model(self):
        """
        Returns model with highest R2 gain
        """

        df = self.compare_baseline_vs_tuned()

        if df is None or df.empty:
            return None

        return df.sort_values("delta_R2", ascending=False).iloc[0].to_dict()
