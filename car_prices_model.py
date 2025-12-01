# Comments for each section can be seen in the included jupyter notebook

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import pickle

RANDOM_STATE=67

cols = ["symboling","normalized-losses","make","fuel-type","aspiration",
        "num-of-doors","body-style","drive-wheels","engine-location",
        "wheel-base","length","width","height","curb-weight","engine-type",
        "num-of-cylinders","engine-size","fuel-system","bore","stroke",
        "compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg",
        "price"]

df = pd.read_csv("data/raw/imports-85.data", names=cols, na_values="?", header=None)
df.head()

numeric_cols = [
    "symboling", "normalized-losses", "wheel-base", "length", "width", "height",
    "curb-weight", "engine-size", "bore", "stroke", "compression-ratio",
    "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[numeric_cols].info()
corr = df[numeric_cols].corr(numeric_only=True)["price"].sort_values(ascending=False)
corr

plt.figure(figsize=(12,8))
sns.heatmap(df[numeric_cols].corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()


important_features = [
    "wheel-base", "length", "width", "curb-weight", "engine-size",
    "horsepower", "city-mpg", "highway-mpg"
]


categorical_features = [
    "make", "fuel-type", "aspiration", "num-of-doors",
    "body-style", "drive-wheels", "engine-location", "engine-type",
    "num-of-cylinders", "fuel-system"
]

selected_features = important_features + categorical_features

df_reduced = df[selected_features + ["price"]].copy()
df_reduced.head()

df_reduced.isna().sum().sort_values(ascending=False)

df_clean = df_reduced.dropna().copy()

print(f"Removed {len(df_reduced) - len(df_clean)} rows; remaining: {len(df_clean)}")

df_clean.isna().sum().any()

df_clean.isna().sum().sort_values(ascending=False)


X = df_clean.drop(columns=["price"])
y = df_clean["price"].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=67
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Column splits
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
print("Numeric:", num_cols)
print("Categorical:", cat_cols)

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ]
)

ridge = Ridge(random_state=RANDOM_STATE)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", ridge),
])

param_grid = {"model__alpha": np.logspace(-3, 3, 13)}
search = GridSearchCV(
    pipe, param_grid=param_grid, cv=5,
    scoring="neg_root_mean_squared_error", n_jobs=-1
)

search.fit(X_train, y_train)

print("Best alpha:", search.best_params_["model__alpha"])
print("Best CV RMSE:", -search.best_score_)

best_model = search.best_estimator_

y_pred_train = best_model.predict(X_train)
y_pred_test  = best_model.predict(X_test)

mse_train = mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
r2_train   = r2_score(y_train, y_pred_train)

mse_test  = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
r2_test   = r2_score(y_test, y_pred_test)

print(f"Train RMSE: {rmse_train:.2f} | R²: {r2_train:.3f}")
print(f" Test RMSE: {rmse_test:.2f} | R²: {r2_test:.3f}")

MODEL_PATH = "car_price_model.pkl"

with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_model, f)

print(f"\nSaved trained model to {MODEL_PATH}")
