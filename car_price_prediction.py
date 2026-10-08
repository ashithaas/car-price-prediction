import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("car data.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nNumber of duplicate rows:", df.duplicated().sum())


# --------------------------------------------------
# 2. DATA PREPROCESSING
# --------------------------------------------------

# Remove duplicate rows
df = df.drop_duplicates()

# Remove Car_Name because it is a high-cardinality identifier
df = df.drop("Car_Name", axis=1)

# Separate features and target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Numerical and categorical features
numerical_features = [
    "Year",
    "Present_Price",
    "Kms_Driven",
    "Owner"
]

categorical_features = [
    "Fuel_Type",
    "Seller_Type",
    "Transmission"
]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# --------------------------------------------------
# 3. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 4. DEFINE MACHINE LEARNING MODELS
# --------------------------------------------------

models = {

    "Linear Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ))
    ]),

    "SVR": Pipeline([
        ("preprocessor", preprocessor),
        ("model", SVR(kernel="rbf"))
    ])
}


# --------------------------------------------------
# 5. TRAIN MODELS
# --------------------------------------------------

print("\nTraining models...")

for name, model in models.items():
    model.fit(X_train, y_train)
    print(name, "trained successfully.")


# --------------------------------------------------
# 6. MAKE PREDICTIONS
# --------------------------------------------------

predictions = {}

for name, model in models.items():
    predictions[name] = model.predict(X_test)


# --------------------------------------------------
# 7. EVALUATE MODELS
# --------------------------------------------------

results = []

for name in models:

    y_pred = predictions[name]

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

results_df = pd.DataFrame(results)

print("\nTest Set Results:")
print(results_df.round(3))


# --------------------------------------------------
# 8. 5-FOLD CROSS-VALIDATION
# --------------------------------------------------

cv_results = []

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_root_mean_squared_error"
    )

    rmse_scores = -scores

    cv_results.append({
        "Model": name,
        "Mean CV RMSE": rmse_scores.mean(),
        "Std CV RMSE": rmse_scores.std()
    })

cv_df = pd.DataFrame(cv_results)

print("\n5-Fold Cross-Validation Results:")
print(cv_df.round(3))


# --------------------------------------------------
# 9. SELECT FINAL MODEL
# --------------------------------------------------

best_model_name = cv_df.loc[
    cv_df["Mean CV RMSE"].idxmin(),
    "Model"
]

print("\nSelected Final Model:", best_model_name)

final_model = models[best_model_name]

final_model.fit(X_train, y_train)

final_predictions = final_model.predict(X_test)


# --------------------------------------------------
# 10. FINAL MODEL EVALUATION
# --------------------------------------------------

final_mae = mean_absolute_error(
    y_test,
    final_predictions
)

final_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        final_predictions
    )
)

final_r2 = r2_score(
    y_test,
    final_predictions
)

print("\nFinal Model Performance:")
print("Model:", best_model_name)
print("MAE:", round(final_mae, 3))
print("RMSE:", round(final_rmse, 3))
print("R2 Score:", round(final_r2, 3))


# --------------------------------------------------
# 11. CREATE PLOTS FOLDER
# --------------------------------------------------

os.makedirs("plots", exist_ok=True)


# --------------------------------------------------
# GRAPH 1: SELLING PRICE DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Selling_Price"],
    bins=20,
    kde=True
)

plt.title("Distribution of Car Selling Prices")
plt.xlabel("Selling Price")
plt.ylabel("Number of Cars")

plt.tight_layout()

plt.savefig(
    "plots/selling_price_distribution.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# GRAPH 2: CORRELATION HEATMAP
# --------------------------------------------------

plt.figure(figsize=(9, 6))

correlation = df.select_dtypes(
    include="number"
).corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Car Price Dataset")

plt.tight_layout()

plt.savefig(
    "plots/correlation_heatmap.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# GRAPH 3: ACTUAL VS PREDICTED
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    final_predictions,
    alpha=0.7
)

min_value = min(
    y_test.min(),
    final_predictions.min()
)

max_value = max(
    y_test.max(),
    final_predictions.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.title("Actual vs Predicted Car Selling Prices")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")

plt.tight_layout()

plt.savefig(
    "plots/actual_vs_predicted.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# GRAPH 4: RESIDUAL PLOT
# --------------------------------------------------

residuals = y_test - final_predictions

plt.figure(figsize=(8, 6))

plt.scatter(
    final_predictions,
    residuals,
    alpha=0.7
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title("Residual Plot for Final Model")
plt.xlabel("Predicted Selling Price")
plt.ylabel("Residuals")

plt.tight_layout()

plt.savefig(
    "plots/residual_plot.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# GRAPH 5: MODEL COMPARISON
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    data=cv_df,
    x="Model",
    y="Mean CV RMSE"
)

plt.title(
    "Model Comparison Based on 5-Fold Cross-Validation RMSE"
)

plt.xlabel("Machine Learning Model")
plt.ylabel("Mean CV RMSE")

plt.xticks(rotation=10)

plt.tight_layout()

plt.savefig(
    "plots/model_comparison_cv_rmse.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 12. SAVE FINAL MODEL
# --------------------------------------------------

joblib.dump(
    final_model,
    "car_price_random_forest.pkl"
)

print("\nFinal model saved successfully!")


# --------------------------------------------------
# 13. FINAL RESULTS TABLE
# --------------------------------------------------

final_results = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R2 Score"
    ],
    "Final Model": [
        final_mae,
        final_rmse,
        final_r2
    ]
})

print("\nFinal Results:")
print(final_results.round(3))

print("\nProject completed successfully! 🚗")