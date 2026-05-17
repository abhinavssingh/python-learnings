import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class LinearModelUtility:

    """
    Beginner-friendly reusable ML utility class.

    Features:
    - Train single / multiple / all models
    - Automatic preprocessing
    - Model evaluation (MSE, R2)
    - Generic hyperparameter tuning
    - Full result tracking via dictionary
    """

    def __init__(self, df, target_col):
        self.df = df
        self.target_col = target_col
        self.results = {}

        # -------------------------------
        # Model registry (central place)
        # -------------------------------
        self.model_registry = {
            "LinearRegression": LinearRegression(),
            "SGDRegressor": SGDRegressor(max_iter=1000, tol=1e-3),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
        }

    # ---------------------------------------------------
    # STEP 1: SPLIT DATA
    # ---------------------------------------------------
    def split_data(self):
        X = self.df.drop(self.target_col, axis=1)
        y = self.df[self.target_col]

        self.num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        self.results['data_split'] = {
            "X_train_shape": self.X_train.shape,
            "X_test_shape": self.X_test.shape,
            "num_columns": self.num_cols,
            "cat_columns": self.cat_cols
        }

    # ---------------------------------------------------
    # STEP 2: PREPROCESSING
    # ---------------------------------------------------
    def build_preprocessor(self, imputer=None, outlier_handler=None):
        steps = []

        # ✅ Step 1: Imputer
        if imputer is not None:
            steps.append(('imputer', imputer))

        # ✅ Step 2: Outlier Handler
        if outlier_handler is not None:
            steps.append(('outlier', outlier_handler))

        # Column-wise transformations
        numeric_transformer = Pipeline([
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        column_transform = ColumnTransformer([
            ('num', numeric_transformer, self.num_cols),
            ('cat', categorical_transformer, self.cat_cols)
        ])

        steps.append(('column_transform', column_transform))

        self.preprocessor = Pipeline(steps)

        self.results['preprocessing'] = {
            "imputer": imputer.get_params() if imputer else "None",
            "outlier_handler": outlier_handler.get_params() if outlier_handler else "None",
            "scaling": "StandardScaler",
            "encoding": "OneHot"
        }

    # ---------------------------------------------------
    # CORE TRAIN FUNCTION
    # ---------------------------------------------------

    def train_model(self, model_name, k_fold=None, use_grid=False, param_grid=None):

        if model_name not in self.model_registry:
            print(f"Model '{model_name}' not found.")
            return

        model = self.model_registry[model_name]

        pipeline = Pipeline([
            ('preprocessing', self.preprocessor),
            ('model', model)
        ])

        # ----------------------------------------------
        # CASE 1: GRID SEARCH (acts like model)
        # ----------------------------------------------
        if use_grid:

            if param_grid is None:
                print("⚠️ param_grid required for GridSearch")
                return

            grid = GridSearchCV(
                pipeline,
                param_grid,
                cv=5,
                scoring='r2',
                n_jobs=-1
            )

            grid.fit(self.X_train, self.y_train)

            self.results[f"{model_name}_GridSearch"] = {
                "mode": "gridsearch",
                "description": self.get_description(model_name) + "\n(GridSearchCV applied)",
                "best_params": grid.best_params_,
                "best_score_cv": grid.best_score_,
                "best_model": grid.best_estimator_
            }

            return

        # ----------------------------------------------
        # CASE 2: K-FOLD CROSS VALIDATION
        # ----------------------------------------------
        if k_fold is not None:

            kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)

            scores = cross_val_score(
                pipeline,
                self.X_train,
                self.y_train,
                cv=kf,
                scoring='r2',
                n_jobs=-1
            )

            self.results[model_name] = {
                "mode": "k-fold",
                "k": k_fold,
                "description": self.get_description(model_name),
                "fold_scores": scores.tolist(),
                "mean_score": np.mean(scores),
                "std_dev": np.std(scores)
            }

            return

        # ----------------------------------------------
        # CASE 3: NORMAL TRAIN-TEST
        # ----------------------------------------------
        pipeline.fit(self.X_train, self.y_train)
        y_pred = pipeline.predict(self.X_test)

        self.results[model_name] = {
            "mode": "train-test",
            "model": str(model),
            "pipeline": pipeline,
            "description": self.get_description(model_name),
            "metrics": {
                "MSE": mean_squared_error(self.y_test, y_pred),
                "R2": r2_score(self.y_test, y_pred)
            }
        }

    # ---------------------------------------------------
    # TRAIN ONE MODEL
    # ---------------------------------------------------
    def train_one(self, model_name, imputer=None, outlier_handler=None):

        if model_name not in self.model_registry:
            print(f"Model '{model_name}' not found.")
            return

        self.split_data()
        self.build_preprocessor(imputer=imputer, outlier_handler=outlier_handler)

        self.train_model(model_name)

        return self.results

    # ---------------------------------------------------
    # TRAIN SELECTED MODELS
    # ---------------------------------------------------
    def train_selected(self, model_list, imputer=None, k_fold=None, outlier_handler=None):

        self.split_data()
        self.build_preprocessor(imputer=imputer, outlier_handler=outlier_handler)

        for model_name in model_list:
            if model_name not in self.model_registry:
                print(f"Skipping invalid model: {model_name}")
                continue

            self.train_model(model_name, k_fold=k_fold)

        return self.results

    # ---------------------------------------------------
    # TRAIN ALL MODELS
    # ---------------------------------------------------

    def train_all(self, imputer=None, k_fold=None, outlier_handler=None):
        """
        Train all models with optional imputer and K-Fold

        Parameters:
        - imputer: CustomImputer instance
        - k_fold: int (e.g., 5) for cross-validation
        """
        return self.train_selected(
            list(self.model_registry.keys()),
            imputer=imputer,
            k_fold=k_fold,
            outlier_handler=outlier_handler
        )

    # ---------------------------------------------------
    # GENERIC HYPERPARAMETER TUNING ✅
    # ---------------------------------------------------

    def tune(self, model_name, param_grid, cv=5, outlier_handler=None):
        """
        Generic tuning method using GridSearchCV

        Example:
        ml.tune("Ridge", {"model__alpha": [0.1, 1, 10]})
        """

        if model_name not in self.model_registry:
            print(f"Model '{model_name}' not found.")
            return

        self.split_data()
        self.build_preprocessor(outlier_handler=outlier_handler)

        model = self.model_registry[model_name]

        pipeline = Pipeline([
            ('preprocessing', self.preprocessor),
            ('model', model)
        ])

        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='r2',
            n_jobs=-1
        )

        grid.fit(self.X_train, self.y_train)

        self.results[f"{model_name}_tuning"] = {
            "best_params": grid.best_params_,
            "best_score": grid.best_score_,
            "best_model": grid.best_estimator_
        }

        return self.results

    # ---------------------------------------------------
    # MODEL DESCRIPTIONS
    # ---------------------------------------------------
    def get_description(self, model_name):

        descriptions = {

            "LinearRegression":
            """Ordinary Least Squares:
Minimizes the squared difference between actual and predicted values.
No regularization → can overfit on large feature spaces.
Best used as baseline model.""",

            "SGDRegressor":
            """Stochastic Gradient Descent:
Updates weights incrementally (one or few samples at a time).
Efficient for large datasets (like your 0.1M rows).
Requires feature scaling.
More sensitive to hyperparameters.""",

            "Ridge":
            """Ridge Regression (L2 Regularization):
Adds penalty = sum of squared coefficients.
Reduces overfitting.
Does NOT eliminate features (keeps all variables).
Works well with multicollinearity.""",

            "Lasso":
            """Lasso Regression (L1 Regularization):
Adds penalty = sum of absolute coefficients.
Can shrink some coefficients to exactly zero.
Performs automatic feature selection.""",

            "ElasticNet":
            """ElasticNet:
Combination of L1 (Lasso) and L2 (Ridge).
Useful when features are highly correlated.
Balances feature selection and stability."""
        }

        return descriptions.get(model_name, "No description available")
