import pandas as pd
import seaborn as seaborn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, cross_val_score


#Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import BayesianRidge


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


import joblib
import os
import json
from functools import partial


MODELS_DIR = "models"

numeric_cols = [
    "condition",
    "cylinders",
    "odometer",
    "car_age",
    "mileage_per_year",
    "days_listed"
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
    "luxury_brand",
    "drive",
    "transmission",
    "size"    
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ]
)

def read_data():
    dataset = pd.read_csv("../cleaned_vehicles.csv")
    
    return dataset

def split_data(dataset: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    
    x = dataset.drop(columns=["price", "id","year"])
    y = np.log1p(dataset["price"])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    
    return x, y, x_train, x_test, y_train, y_test
    
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

def error_analysis(rf_model, x_test, y_test):
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

def evaluate_model(model, x_test, y_test):
    pred_log = model.predict(x_test)
    pred = np.expm1(pred_log)
    y_test_real = np.expm1(y_test)

    mae = mean_absolute_error(y_test_real, pred)
    rmse = np.sqrt(mean_squared_error(y_test_real, pred))
    r2 = r2_score(y_test_real, pred)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    return mae, rmse, r2, pred

def evaluate_predictions(pred, y_test):
    y_test_real = np.expm1(y_test)
    
    mae = mean_absolute_error(y_test_real, pred)
    rmse = np.sqrt(mean_squared_error(y_test_real, pred))
    r2 = r2_score(y_test_real, pred)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")

    return mae, rmse, r2, pred

def cross_validate_model(model, X, y, n_splits=3):
    if hasattr(model.named_steps["model"], "early_stopping_rounds"):
        model.named_steps["model"].early_stopping_rounds = None
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=-1)
    
    print(f"CV R²: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores

def save_model_results(x_train, y_train, x_test, y_test, *models: list[callable]):
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    sample_cars = x_test.sample(n=30, random_state=42)
    sample_prices = y_test.loc[sample_cars.index]    
    
    result_dir = {}
    
    result_dir["sample_cars"] = {
        "actual_prices": [round(p, 2) for p in np.expm1(sample_prices).tolist()],
        "car_indices": sample_cars.index.tolist()
    }
    
    model_dict = {}
    
    for model_fn in models:
        if "xgboost" in (model_fn.func.__name__ if isinstance(model_fn, partial) else model_fn.__name__):
            _, result_dict = model_fn(x_train, y_train, x_test, y_test, sample_cars, sample_prices)
        else:
            _, result_dict = model_fn(x_train, y_train, sample_cars, sample_prices)
        name = model_fn.func.__name__ if isinstance(model_fn, partial) else model_fn.__name__
        name = name.replace("_pipeline", "").replace("_", " ").capitalize()
        model_dict[name] = result_dict
        
    result_dir["models"] = model_dict
    
    with open("model_results.json", "w") as f:
        json.dump(result_dir, f, indent=4)

def linear_regression_pipeline(x_train, y_train, x_test, y_test):
    
    if os.path.exists(os.path.join(MODELS_DIR, "linear_model.joblib")):
        model = joblib.load(os.path.join(MODELS_DIR, "linear_model.joblib"))
    else:
        model = Pipeline ([
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ])
    
        print("Fitting Linear Regression...")
        model.fit(x_train, y_train)

    print("Evaluating Linear Regression...")
    mae, rmse, r2, pred = evaluate_model(model, x_test, y_test)
    
    joblib.dump(model, os.path.join(MODELS_DIR, "linear_model.joblib"))
    
    return model, {
                   "mae": mae,
                   "rmse": rmse,
                   "r2": r2,
                   "predictions": pred.tolist(),
                   "upper_bound": None,
                   "lower_bound": None
                   }
    
def random_forest_pipeline(x_train, y_train, x_test, y_test, x, y):
    
    if os.path.exists(os.path.join(MODELS_DIR, "random_forest_model.joblib")):
        model = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.joblib"))
    else:
        print("Fitting Random Forest...")
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=100,
                max_depth = 10,
                random_state=42,
                n_jobs=-1
            ))
            ])

        model.fit(x_train, y_train)

    print("Evaluating Random Forest...")
    mae, rmse, r2, pred = evaluate_model(model, x_test, y_test)
    
    print("Cross-validating Random Forest...")
    #cross_validate_model(model, x, y,  n_splits=3)
    
    return model, {
                   "mae": mae,
                   "rmse": rmse,
                   "r2": r2,
                   "predictions": pred.tolist(),
                   "upper_bound": None,
                   "lower_bound": None
                   }

def random_forest_interval_pipeline(rf_model, sample, real_price):

    x_processed = rf_model.named_steps["preprocessor"].transform(sample)

    forest = rf_model.named_steps["model"]

    tree_predictions = []

    for tree in forest.estimators_:

        pred_log = tree.predict(x_processed)

        # Convert from log(price)
        pred = np.expm1(pred_log)

        tree_predictions.append(pred)

    tree_predictions = np.array(tree_predictions).flatten()

    mean_pred = np.mean(tree_predictions)
    lower_bound = np.percentile(tree_predictions, 5)
    upper_bound = np.percentile(tree_predictions, 95)

    real_price = np.expm1(real_price)

    print("Prediction:")
    print(f"Estimated price: ${mean_pred:,.0f}")

    print("\nPrediction interval:")
    print(f"${lower_bound:,.0f} - ${upper_bound:,.0f}")

    print("\nReal price:")
    print(f"${real_price:,.0f}\n\n")

    return mean_pred, lower_bound, upper_bound
    
def xgboost_pipeline(x_train, y_train, x_test, y_test,sample_cars, sample_prices, x, y):
    
    if os.path.exists(os.path.join(MODELS_DIR, "xgboost_model.joblib")):
        model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.joblib"))
    else :
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBRegressor(
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                early_stopping_rounds=50,
            ))
            ])
        
        model.named_steps["preprocessor"].fit(x_train)

        x_test_transformed = model.named_steps["preprocessor"].transform(x_test)

        model.fit(
            x_train, y_train,
            model__eval_set=[(x_test_transformed, y_test)],
            model__verbose=100,
        )
    
    print("Evaluating XGBoost...")
    # full test metrics for honest reporting
    mae, rmse, r2, _ = evaluate_model(model, x_test, y_test)

    # sample predictions for dashboard chart
    _, _, _, pred = evaluate_model(model, sample_cars, sample_prices)
    
    print("Cross-validating XGBoost...")
    #cross_validate_model(model, x, y, n_splits=3)
    
    return model, {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "predictions": pred.tolist(),
        "upper_bound": None,
        "lower_bound": None
    }
    
def bayesian_ridge_pipeline(x_train, y_train, x_test, y_test):
    if os.path.exists(os.path.join(MODELS_DIR, "bayesian_ridge_model.joblib")):
        model = joblib.load(os.path.join(MODELS_DIR, "bayesian_ridge_model.joblib"))
    else:    
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("model", BayesianRidge())
        ])
        
        print("Fitting Bayesian Ridge...")
        model.fit(x_train, y_train)
        
    pred_mean, pred_std = model.named_steps["model"].predict(
    model.named_steps["preprocessor"].transform(x_test),
    return_std=True
    )
    
    print("Evaluating Bayesian Ridge...")
    mae, rmse, r2, pred = evaluate_model(model, x_test, y_test)
    
    print("\nSample prediction 30 cars:")
    lower = np.expm1(pred_mean - pred_std)
    upper = np.expm1(pred_mean + pred_std)
    price = np.expm1(pred_mean)
    
    for i in range(30):
        print(f"Car {i+1}: ${price[i]:,.0f}  range: ${lower[i]:,.0f} – ${upper[i]:,.0f}  (actual: ${np.expm1(y_test.iloc[i]):,.0f})")    
    
    
    return model, {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "predictions": pred.tolist(),
        "upper_bound": upper.tolist(),
        "lower_bound": lower.tolist()
    }

def gaussian_process_pipeline(x_train, y_train, x_test, y_test):
    x_train_sample = x_train[numeric_cols].sample(n=3000, random_state=42)
    y_train_sample = y_train.loc[x_train_sample.index]
    x_test_gp = x_test[numeric_cols]
    
    imputer = SimpleImputer(strategy="median")
    x_train_processed = imputer.fit_transform(x_train_sample)
    x_test_processed = imputer.transform(x_test_gp)
    
    if os.path.exists(os.path.join(MODELS_DIR, "gaussian_process_model.joblib")):
        gp = joblib.load(os.path.join(MODELS_DIR, "gaussian_process_model.joblib"))
    else:
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e8)) + WhiteKernel(noise_level=1.0)
        
        gp = GaussianProcessRegressor(
            kernel=kernel,
            random_state=42,
            n_restarts_optimizer=3
        )
        
        print("Fitting Gaussian Process...")
        gp = gp.fit(x_train_processed, y_train_sample)
    
    print("Getting predictions...")
    pred_result = gp.predict(x_test_processed, return_std=True)
    pred_mean = pred_result[0]
    pred_std = pred_result[1]
    
    print("Evaluating Gaussian Process...")
    mae, rmse, r2, pred = evaluate_predictions(pred_mean, y_test)
    
    print("\nSample prediction 30 cars:")
    lower = np.expm1(pred_mean - pred_std)
    upper = np.expm1(pred_mean + pred_std)
    price = np.expm1(pred_mean)
    
    for i in range(30):
        print(f"Car {i+1}: ${price[i]:,.0f}  range: ${lower[i]:,.0f} – ${upper[i]:,.0f}  (actual: ${np.expm1(y_test.iloc[i]):,.0f})")    
    
    return gp, {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "predictions": pred.tolist(),
        "upper_bound": upper.tolist(),
        "lower_bound": lower.tolist()
    }

def quantile_regression_pipeline(x_train, y_train, x_test, y_test):
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)
    
    if os.path.exists(os.path.join(MODELS_DIR, "quantile_regression_model_upper.joblib")) and \
       os.path.exists(os.path.join(MODELS_DIR, "quantile_regression_model_median.joblib")) and \
       os.path.exists(os.path.join(MODELS_DIR, "quantile_regression_model_lower.joblib")):
        lower_model = joblib.load(os.path.join(MODELS_DIR, "quantile_regression_model_lower.joblib"))
        median_model = joblib.load(os.path.join(MODELS_DIR, "quantile_regression_model_median.joblib"))
        upper_model = joblib.load(os.path.join(MODELS_DIR, "quantile_regression_model_upper.joblib"))
    else:
        models = {}
        
        for alpha, name in [(0.1, "lower"), (0.5, "median"), (0.9, "upper")]:
            print(f"Fitting {name} quantile model...")
            models[name] = XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=alpha,
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                n_jobs=-1,  
                random_state=42
            )
            
            models[name].fit(x_train_processed, y_train)
        
        
        lower_model = models["lower"]
        median_model = models["median"]
        upper_model = models["upper"]
    
    print("Getting predictions...")
    lower_pred_log = lower_model.predict(x_test_processed)
    lower_pred = np.expm1(lower_pred_log)
    
    median_pred_log = median_model.predict(x_test_processed)
    median_pred = np.expm1(median_pred_log)
    
    upper_pred_log = upper_model.predict(x_test_processed)
    upper_pred = np.expm1(upper_pred_log)
    
    print("Evaluating Quantile Regression models...")
    mae, rmse, r2, pred = evaluate_predictions(median_pred, y_test)
    
    print("\nSample prediction 30 cars:")
    for i in range(30):
        print(f"Car {i+1}: ${median_pred[i]:,.0f}  range: ${lower_pred[i]:,.0f} – ${upper_pred[i]:,.0f}  (actual: ${np.expm1(y_test.iloc[i]):,.0f})")
    
    return median_model, {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "predictions": pred.tolist(),
        "upper_bound": upper_pred.tolist(),
        "lower_bound": lower_pred.tolist()
    }

def main():
    dataset = read_data()
    
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    
    save_model_results(
        x_train, y_train, x_test, y_test,
        linear_regression_pipeline,
        partial(random_forest_pipeline, x=x, y=y),
        partial(xgboost_pipeline, x=x, y=y),
        bayesian_ridge_pipeline,
        gaussian_process_pipeline,
        quantile_regression_pipeline
        )
    
if __name__ == '__main__':
    main()