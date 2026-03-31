from preprocessing import *
from model import LinearRegressionScratch
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression, Ridge


# -------- SAVE FUNCTION --------
def save_model(model, mean, std, path="model.pkl"):
    with open(path, "wb") as f:
        pickle.dump({
            "model": model,
            "mean": mean,
            "std": std
        }, f)


# -------- LOAD + PREPROCESS --------
df = load_data("housing_price_dataset.csv")

df = handle_missing(df)
df = feature_engineering(df)
df = encode_features(df)

print("Columns:", df.columns)


# -------- SPLIT --------
X, y = split_feature_target(df)
X_train, X_test, y_train, y_test = train_test_split(X, y)


# -------- SCALE --------
X_train, mean, std = scale_features(X_train)
X_test = (X_test - mean) / std


# -------- YOUR MODEL --------
model = LinearRegressionScratch()
model.fit(X_train, y_train)

preds = model.predict(X_test)

rmse = np.sqrt(np.mean((preds - y_test) ** 2))
mae = np.mean(np.abs(preds - y_test))

ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_res = np.sum((y_test - preds) ** 2)
r2 = 1 - (ss_res / ss_total)

print("\n--- YOUR MODEL ---")
print("RMSE:", rmse)
print("MAE :", mae)
print("R2  :", r2)


# -------- SKLEARN --------
sk_model = LinearRegression()
sk_model.fit(X_train, y_train)

sk_preds = sk_model.predict(X_test)

sk_rmse = np.sqrt(np.mean((sk_preds - y_test) ** 2))
print("\n--- SKLEARN ---")
print("RMSE:", sk_rmse)


# -------- RIDGE --------
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

ridge_preds = ridge.predict(X_test)
ridge_rmse = np.sqrt(np.mean((ridge_preds - y_test) ** 2))

print("\n--- RIDGE ---")
print("RMSE:", ridge_rmse)


# -------- SAVE MODEL --------
save_model(model, mean, std)
print("\nModel saved as model.pkl")
