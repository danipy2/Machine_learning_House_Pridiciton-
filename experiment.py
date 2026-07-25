


import pandas as pd
import numpy as np
import time
import joblib
import os

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    MinMaxScaler,
    StandardScaler
)
from sklearn.impute import SimpleImputer

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

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)



url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

df = pd.read_csv(url)



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



target = "median_house_value"

X = df.drop(columns=[target])

y = df[target]




X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)



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



encoders = {

    "Label Encoding": OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ),

    "One-Hot Encoding": OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
}



scalers = {

    "No Scaling": "passthrough",

    "Min-Max Scaling": MinMaxScaler(),

    "Standardization": StandardScaler()
}



models = {

    "SVR": SVR(
        kernel="rbf",
        C=100,
        gamma="scale"
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    ),

    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(
        alpha=1.0
    ),

    "Lasso Regression": Lasso(
        alpha=0.1,
        max_iter=10000
    ),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    )
}



results = []

experiment_number = 1


best_rmse = float("inf")

best_pipeline = None

best_configuration = None



for encoding_name, encoder in encoders.items():

    for scaling_name, scaler in scalers.items():

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



            numerical_pipeline = Pipeline([

                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),

                (
                    "scaler",
                    scaler
                )
            ])



            categorical_pipeline = Pipeline([

                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),

                (
                    "encoder",
                    encoder
                )
            ])



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



            start_time = time.time()

            pipeline.fit(
                X_train,
                y_train
            )

            training_time = (
                time.time() - start_time
            )



            y_pred = pipeline.predict(
                X_test
            )



            mae = mean_absolute_error(
                y_test,
                y_pred
            )



            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    y_pred
                )
            )



            r2 = r2_score(
                y_test,
                y_pred
            )



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



            if rmse < best_rmse:

                best_rmse = rmse

                best_pipeline = pipeline

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


            experiment_number += 1



results_df = pd.DataFrame(
    results
)



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



best_model_rmse = results_df.loc[
    results_df["RMSE"].idxmin()
]

best_model_mae = results_df.loc[
    results_df["MAE"].idxmin()
]

best_model_r2 = results_df.loc[
    results_df["R2"].idxmax()
]



print("\n\n")

print("=" * 100)

print(
    "BEST MODEL ACCORDING TO RMSE"
)

print("=" * 100)

print(
    best_model_rmse.to_string()
)



print("\n\n")

print("=" * 100)

print(
    "BEST MODEL ACCORDING TO MAE"
)

print("=" * 100)

print(
    best_model_mae.to_string()
)



print("\n\n")

print("=" * 100)

print(
    "BEST MODEL ACCORDING TO R2"
)

print("=" * 100)

print(
    best_model_r2.to_string()
)



results_df.to_csv(
    "housing_42_experiments.csv",
    index=False
)



results_by_rmse.to_csv(
    "housing_42_results_sorted_by_rmse.csv",
    index=False
)




os.makedirs(
    "model",
    exist_ok=True
)



model_path = (
    "model/"
    "best_house_price_model.joblib"
)


joblib.dump(
    best_pipeline,
    model_path
)



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