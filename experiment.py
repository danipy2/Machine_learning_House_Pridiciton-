# ============================================================
# HOUSE PRICE PREDICTION - 42 EXPERIMENTS
# ============================================================
#
# 2 Encoding Methods
#   1. Label/Ordinal Encoding
#   2. One-Hot Encoding
#
# 3 Scaling Methods
#   1. No Scaling
#   2. Min-Max Scaling
#   3. Standardization
#
# 7 Regression Algorithms
#   1. SVR
#   2. Random Forest Regressor
#   3. Gradient Boosting Regressor
#   4. Linear Regression
#   5. Ridge Regression
#   6. Lasso Regression
#   7. Decision Tree Regressor
#
# Total:
# 2 x 3 x 7 = 42 Experiments
#
# The best-performing complete pipeline is automatically
# saved as:
#
# best_house_price_model.joblib
#
# The best model is selected based on the lowest RMSE.
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import time
import joblib
import os

# Model selection
from sklearn.model_selection import train_test_split

# Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    MinMaxScaler,
    StandardScaler
)
from sklearn.impute import SimpleImputer

# Regression models
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)
from sklearn.tree import DecisionTreeRegressor

# Evaluation metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

df = pd.read_csv(url)


# ============================================================
# 3. BASIC DATASET INFORMATION
# ============================================================

print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 4. DEFINE TARGET AND FEATURES
# ============================================================

# Target variable
target = "median_house_value"

# X = input features
X = df.drop(columns=[target])

# y = target variable
y = df[target]


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

# Use the same split for all 42 experiments
# This makes the comparison fair.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)


# ============================================================
# 6. DEFINE NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numerical_features = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income"
]

categorical_features = [
    "ocean_proximity"
]


# ============================================================
# 7. DEFINE ENCODING METHODS
# ============================================================

encoders = {

    # Label/Ordinal Encoding
    "Label Encoding": OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ),

    # One-Hot Encoding
    "One-Hot Encoding": OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
}


# ============================================================
# 8. DEFINE SCALING METHODS
# ============================================================

scalers = {

    # No scaling
    "No Scaling": "passthrough",

    # Min-Max Scaling
    "Min-Max Scaling": MinMaxScaler(),

    # Standardization
    "Standardization": StandardScaler()
}


# ============================================================
# 9. DEFINE 7 REGRESSION MODELS
# ============================================================

models = {

    # 1. Support Vector Regression
    "SVR": SVR(
        kernel="rbf",
        C=100,
        gamma="scale"
    ),

    # 2. Random Forest Regressor
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    # 3. Gradient Boosting Regressor
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    ),

    # 4. Linear Regression
    "Linear Regression": LinearRegression(),

    # 5. Ridge Regression
    "Ridge Regression": Ridge(
        alpha=1.0
    ),

    # 6. Lasso Regression
    "Lasso Regression": Lasso(
        alpha=0.1,
        max_iter=10000
    ),

    # 7. Decision Tree Regressor
    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    )
}


# ============================================================
# 10. INITIALIZE EXPERIMENT VARIABLES
# ============================================================

results = []

experiment_number = 1

# ------------------------------------------------------------
# Variables used to track the best model
# ------------------------------------------------------------

# Start with infinity because lower RMSE is better
best_rmse = float("inf")

# Will contain the complete trained pipeline
best_pipeline = None

# Will contain information about the best configuration
best_configuration = None


# ============================================================
# 11. RUN ALL 42 EXPERIMENTS
# ============================================================

# Loop through 2 encoding methods
for encoding_name, encoder in encoders.items():

    # Loop through 3 scaling methods
    for scaling_name, scaler in scalers.items():

        # Loop through 7 machine learning models
        for model_name, model in models.items():

            print("\n" + "=" * 70)

            print(
                f"EXPERIMENT {experiment_number} / 42"
            )

            print("=" * 70)

            print(
                "Encoding :",
                encoding_name
            )

            print(
                "Scaling  :",
                scaling_name
            )

            print(
                "Model    :",
                model_name
            )


            # ====================================================
            # NUMERICAL PIPELINE
            # ====================================================

            numerical_pipeline = Pipeline([

                # Fill missing numerical values with median
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),

                # Apply selected scaling method
                (
                    "scaler",
                    scaler
                )
            ])


            # ====================================================
            # CATEGORICAL PIPELINE
            # ====================================================

            categorical_pipeline = Pipeline([

                # Fill missing categorical values
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),

                # Apply selected encoding method
                (
                    "encoder",
                    encoder
                )
            ])


            # ====================================================
            # COMBINE NUMERICAL AND CATEGORICAL PIPELINES
            # ====================================================

            preprocessor = ColumnTransformer([

                (
                    "numerical",
                    numerical_pipeline,
                    numerical_features
                ),

                (
                    "categorical",
                    categorical_pipeline,
                    categorical_features
                )
            ])


            # ====================================================
            # CREATE COMPLETE PIPELINE
            # ====================================================

            pipeline = Pipeline([

                (
                    "preprocessing",
                    preprocessor
                ),

                (
                    "model",
                    model
                )
            ])


            # ====================================================
            # TRAIN MODEL
            # ====================================================

            start_time = time.time()

            pipeline.fit(
                X_train,
                y_train
            )

            training_time = (
                time.time() - start_time
            )


            # ====================================================
            # MAKE PREDICTIONS
            # ====================================================

            y_pred = pipeline.predict(
                X_test
            )


            # ====================================================
            # CALCULATE MAE
            # ====================================================

            mae = mean_absolute_error(
                y_test,
                y_pred
            )


            # ====================================================
            # CALCULATE RMSE
            # ====================================================

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    y_pred
                )
            )


            # ====================================================
            # CALCULATE R2 SCORE
            # ====================================================

            r2 = r2_score(
                y_test,
                y_pred
            )


            # ====================================================
            # STORE RESULTS
            # ====================================================

            results.append({

                "Experiment": experiment_number,

                "Encoding": encoding_name,

                "Scaling": scaling_name,

                "Model": model_name,

                "MAE": mae,

                "RMSE": rmse,

                "R2": r2,

                "Training Time (sec)": training_time

            })


            # ====================================================
            # CHECK IF THIS IS THE BEST MODEL
            # ====================================================

            if rmse < best_rmse:

                # Update best RMSE
                best_rmse = rmse

                # Save the complete pipeline
                best_pipeline = pipeline

                # Save configuration information
                best_configuration = {

                    "Experiment": experiment_number,

                    "Encoding": encoding_name,

                    "Scaling": scaling_name,

                    "Model": model_name,

                    "MAE": mae,

                    "RMSE": rmse,

                    "R2": r2

                }

                print("\n*** NEW BEST MODEL FOUND ***")

                print(
                    "Encoding:",
                    encoding_name
                )

                print(
                    "Scaling:",
                    scaling_name
                )

                print(
                    "Model:",
                    model_name
                )

                print(
                    "RMSE:",
                    rmse
                )


            # Move to next experiment
            experiment_number += 1


# ============================================================
# 12. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# 13. DISPLAY ALL 42 RESULTS
# ============================================================

print("\n\n")

print("=" * 100)

print(
    "ALL 42 EXPERIMENT RESULTS"
)

print("=" * 100)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 14. SORT BY RMSE
# ============================================================

# Lower RMSE is better

results_by_rmse = results_df.sort_values(
    by="RMSE",
    ascending=True
)


print("\n\n")

print("=" * 100)

print(
    "TOP 10 MODELS BY RMSE"
)

print("=" * 100)

print(
    results_by_rmse.head(10).to_string(
        index=False
    )
)


# ============================================================
# 15. SORT BY MAE
# ============================================================

# Lower MAE is better

results_by_mae = results_df.sort_values(
    by="MAE",
    ascending=True
)


print("\n\n")

print("=" * 100)

print(
    "TOP 10 MODELS BY MAE"
)

print("=" * 100)

print(
    results_by_mae.head(10).to_string(
        index=False
    )
)


# ============================================================
# 16. SORT BY R2
# ============================================================

# Higher R2 is better

results_by_r2 = results_df.sort_values(
    by="R2",
    ascending=False
)


print("\n\n")

print("=" * 100)

print(
    "TOP 10 MODELS BY R2 SCORE"
)

print("=" * 100)

print(
    results_by_r2.head(10).to_string(
        index=False
    )
)


# ============================================================
# 17. FIND THE BEST MODEL ACCORDING TO EACH METRIC
# ============================================================

best_model_rmse = results_df.loc[
    results_df["RMSE"].idxmin()
]

best_model_mae = results_df.loc[
    results_df["MAE"].idxmin()
]

best_model_r2 = results_df.loc[
    results_df["R2"].idxmax()
]


# ============================================================
# 18. DISPLAY BEST MODEL ACCORDING TO RMSE
# ============================================================

print("\n\n")

print("=" * 100)

print(
    "BEST MODEL ACCORDING TO RMSE"
)

print("=" * 100)

print(
    best_model_rmse.to_string()
)


# ============================================================
# 19. DISPLAY BEST MODEL ACCORDING TO MAE
# ============================================================

print("\n\n")

print("=" * 100)

print(
    "BEST MODEL ACCORDING TO MAE"
)

print("=" * 100)

print(
    best_model_mae.to_string()
)


# ============================================================
# 20. DISPLAY BEST MODEL ACCORDING TO R2
# ============================================================

print("\n\n")

print("=" * 100)

print(
    "BEST MODEL ACCORDING TO R2"
)

print("=" * 100)

print(
    best_model_r2.to_string()
)


# ============================================================
# 21. SAVE ALL RESULTS TO CSV
# ============================================================

results_df.to_csv(
    "housing_42_experiments.csv",
    index=False
)


# ============================================================
# 22. SAVE SORTED RESULTS TO CSV
# ============================================================

results_by_rmse.to_csv(
    "housing_42_results_sorted_by_rmse.csv",
    index=False
)


# ============================================================
# 23. SAVE THE BEST COMPLETE PIPELINE
# ============================================================

# Create model directory if it doesn't exist

os.makedirs(
    "model",
    exist_ok=True
)


# Save the complete pipeline
# This includes:
#
# 1. Missing value imputation
# 2. Encoding
# 3. Scaling
# 4. Trained regression model

model_path = (
    "model/"
    "best_house_price_model.joblib"
)


joblib.dump(
    best_pipeline,
    model_path
)


# ============================================================
# 24. DISPLAY SAVED MODEL INFORMATION
# ============================================================

print("\n\n")

print("=" * 100)

print(
    "BEST MODEL SAVED SUCCESSFULLY"
)

print("=" * 100)

print(
    "\nModel file:"
)

print(
    model_path
)


print(
    "\nBest model configuration:"
)

for key, value in best_configuration.items():

    print(
        f"{key}: {value}"
    )


# ============================================================
# 25. FINAL SUMMARY
# ============================================================

print("\n\n")

print("=" * 100)

print(
    "EXPERIMENT COMPLETE"
)

print("=" * 100)

print(
    "\nTotal experiments completed:",
    len(results_df)
)

print(
    "\nResults saved to:"
)

print(
    "1. housing_42_experiments.csv"
)

print(
    "2. housing_42_results_sorted_by_rmse.csv"
)

print(
    "3. model/best_house_price_model.joblib"
)

print(
    "\nThe best complete preprocessing + model pipeline"
)

print(
    "has been automatically saved for deployment."
)