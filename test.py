from preprocessing import *
from model import LinearRegressionScratch

df = load_data("housing_price_dataset.csv")

df = handle_missing(df)
df = encode_features(df)
df = feature_engineering(df)

X,y = split_feature_target(df)

X_train, X_test, y_train, y_test = train_test_split(X,y)

X_train, mean, std = scale_features(X_train)
X_test = (X_test - mean) / std

#print("Train Shape", X_train.shape)
#print("Test Shape", X_test.shape)

#print("Columns after preprocessing:", df.columns)

#print("First row (scaled):", X_train[0])
#print("Mean approx:", X_train.mean(axis=0))
#print("Std approx:", X_train.std(axis=0))

model = LinearRegressionScratch(lr=0.005, epochs=5000)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print(preds[:5])
print("Actual:", y_test[:5])
print("Predicted:", preds[:5])
rmse = np.sqrt(np.mean((preds - y_test) ** 2))
print("RMSE:", rmse)
print(model.losses[:5])
print(model.losses[-5:])
preds = model.predict(X_test)

# MAE
mae = np.mean(np.abs(preds - y_test))
print("MAE:", mae)

# R2 Score
ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_residual = np.sum((y_test - preds) ** 2)
r2 = 1 - (ss_residual / ss_total)
print("R2 Score:", r2)