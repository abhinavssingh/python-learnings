from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class LinearModelUtility:

    """
    A beginner-friendly class to train multiple Linear Models and inspect internal outputs.

    Supports:
    - Linear Regression
    - SGD Regressor
    - Ridge Regression (L2)
    - Lasso Regression (L1)
    - ElasticNet (L1 + L2)
    """

    def __init__(self, df, target_col):
        self.df = df
        self.target_col = target_col
        self.results = {}

    # ---------------------------------------------------
    # STEP 1: SPLIT DATA
    # ---------------------------------------------------
    def split_data(self):
        X = self.df.drop(self.target_col, axis=1)
        y = self.df[self.target_col]

        self.num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.cat_cols = X.select_dtypes(include=['object']).columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        self.results['data_split'] = {
            "X_train_shape": X_train.shape,
            "X_test_shape": X_test.shape,
            "num_columns": self.num_cols,
            "cat_columns": self.cat_cols
        }

        return X_train, X_test, y_train, y_test

    # ---------------------------------------------------
    # STEP 2: PREPROCESSING
    # ---------------------------------------------------
    def build_preprocessor(self):
        numeric_transformer = Pipeline([
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, self.num_cols),
            ('cat', categorical_transformer, self.cat_cols)
        ])

        self.results['preprocessing'] = {
            "scaling": "StandardScaler applied to numerical columns",
            "encoding": "OneHotEncoder applied to categorical columns",
            "total_features_after_encoding": "increases after one-hot encoding"
        }

        return preprocessor

    # ---------------------------------------------------
    # COMMON TRAIN FUNCTION
    # ---------------------------------------------------
    def train_model(self, model_name, model, X_train, X_test, y_train, y_test, preprocessor):
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('model', model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        self.results[model_name] = {
            "model": str(model),
            "description": self.get_description(model_name),
            "metrics": {
                "MSE": mean_squared_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred)
            }
        }

    # ---------------------------------------------------
    # MODEL DESCRIPTIONS (VERY IMPORTANT FOR LEARNING)
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

    # ---------------------------------------------------
    # TRAIN ALL MODELS
    # ---------------------------------------------------
    def train_all(self):
        X_train, X_test, y_train, y_test = self.split_data()
        preprocessor = self.build_preprocessor()

        self.train_model("LinearRegression", LinearRegression(),
                         X_train, X_test, y_train, y_test, preprocessor)

        self.train_model("SGDRegressor", SGDRegressor(max_iter=1000, tol=1e-3),
                         X_train, X_test, y_train, y_test, preprocessor)

        self.train_model("Ridge", Ridge(alpha=1.0),
                         X_train, X_test, y_train, y_test, preprocessor)

        self.train_model("Lasso", Lasso(alpha=0.1),
                         X_train, X_test, y_train, y_test, preprocessor)

        self.train_model("ElasticNet", ElasticNet(alpha=0.1, l1_ratio=0.5),
                         X_train, X_test, y_train, y_test, preprocessor)

        return self.results

    # ---------------------------------------------------
    # GRID SEARCH (ADVANCED STEP)
    # ---------------------------------------------------
    def tune_ridge(self):
        X_train, X_test, y_train, y_test = self.split_data()
        preprocessor = self.build_preprocessor()

        param_grid = {
            'model__alpha': [0.01, 0.1, 1, 10]
        }

        grid = GridSearchCV(
            Pipeline([
                ('preprocessing', preprocessor),
                ('model', Ridge())
            ]),
            param_grid,
            cv=5,
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        self.results['Ridge_GridSearch'] = {
            "best_params": grid.best_params_,
            "best_score": grid.best_score_
        }

        return self.results
