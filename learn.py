import pandas as pd
import seaborn as seaborn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import joblib



numeric_cols = [
    "condition",
    "cylinders",
    "odometer",
    "car_age",
    "mileage_per_year"
]

categorical_cols = [
    "region",
    "manufacturer",
    "fuel",
    "type",
    "paint_color",
    "state",
    "title_status",
    "vin_present",
    "model_clean",
    "luxury_brand"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

def read_data():
    dataset = pd.read_csv("../cleaned_vehicles.csv")
    
    return dataset

def split_data(dataset: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    
    X = dataset.drop(columns=["price", "id","year"])
    y = np.log1p(dataset["price"])

    global x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
def feature_importance(model):
    rf = model.named_steps["model"]
    importances = rf.feature_importances_
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    
    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    feat_imp = feat_imp.sort_values("importance", ascending=False)

    print(feat_imp.head(20))
    
    return feat_imp

def error_analysis(rf_model):
    pred_log = rf_model.predict(x_test)
    pred = np.expm1(pred_log)
    y_test_real = np.expm1(y_test)

    results = x_test.copy()
    results["actual"] = y_test_real
    results["pred"] = pred
    results["error"] = abs(results["actual"] - results["pred"])

    print(results.sort_values("error", ascending=False).head(10))
    print(results.groupby("manufacturer")["error"].mean().sort_values(ascending=False).head(10))
    print(results.groupby("type")["error"].mean())
    print(results.groupby(pd.cut(results["car_age"], bins=5))["error"].mean())
    return results

def linear_regression_pipeline():
    model = Pipeline ([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])
    
    model.fit(x_train, y_train)
    
    pred_log = model.predict(x_test)
    
    pred = np.expm1(pred_log)

    y_test_real = np.expm1(y_test)

    mae = mean_absolute_error(y_test_real, pred)
    rmse = np.sqrt(mean_squared_error(y_test_real, pred))
    r2 = r2_score(y_test_real, pred)

    print("Linear Regression Performance:")
    print(f"MAE: {mae}, RMSE: {rmse}, R²: {r2}")
    
def random_forest_pipeline():
    print("Training Random Forest...")
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=100,
            max_depth = 20,
            random_state=42,
            n_jobs=-1
        ))
        ])

    model.fit(x_train, y_train)

    pred_log = model.predict(x_test)
    pred = np.expm1(pred_log)

    y_test_real = np.expm1(y_test)

    print("Evaluating Random Forest...")
    mae = mean_absolute_error(y_test_real, pred)
    rmse = np.sqrt(mean_squared_error(y_test_real, pred))
    r2 = r2_score(y_test_real, pred)

    print("Random Forest Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    return model

def random_forest_interval_pipeline(rf_model, sample, real_price):

    # Preprocess sample
    x_processed = rf_model.named_steps["preprocessor"].transform(sample)

    # Get forest
    forest = rf_model.named_steps["model"]

    tree_predictions = []

    # Predict with every tree
    for tree in forest.estimators_:

        pred_log = tree.predict(x_processed)

        # Convert from log(price)
        pred = np.expm1(pred_log)

        tree_predictions.append(pred)

    # Convert list -> numpy array
    tree_predictions = np.array(tree_predictions).flatten()

    # Average prediction
    mean_pred = np.mean(tree_predictions)

    # Prediction interval
    lower_bound = np.percentile(tree_predictions, 5)
    upper_bound = np.percentile(tree_predictions, 95)

    # Convert real value from log(price)
    real_price = np.expm1(real_price)

    print("Prediction:")
    print(f"Estimated price: ${mean_pred:,.0f}")

    print("\nPrediction interval:")
    print(f"${lower_bound:,.0f} - ${upper_bound:,.0f}")

    print("\nReal price:")
    print(f"${real_price:,.0f}\n\n")

    return mean_pred, lower_bound, upper_bound
    
    
    

def main():
    dataset = read_data()
    
    split_data(dataset)
    
    #linear_regression_pipeline()
    
    model = joblib.load("random_forest_model.pkl")
    
    for i in range(30):
        sample_index = i

        sample = x_test.iloc[[sample_index]]

        real_price = y_test.iloc[sample_index]

        random_forest_interval_pipeline(
            model,
            sample,
            real_price)
    
    
    
if __name__ == '__main__':
    main()