"""
Wine Quality Prediction - Mini Project (Data Science)
Dataset : UCI Wine Quality (red + white vinho verde wines)
Tasks   : 1) Regression  -> predict the quality score (0-10)
          2) Classification -> predict Good (quality >= 6) vs Not Good
Run     : python wine_quality.py
"""
import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

RANDOM_STATE = 42
DATA_DIR, OUT_DIR = "data", "outputs"
BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
FILES = {"red": "winequality-red.csv", "white": "winequality-white.csv"}

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------- 1. LOAD DATA
def load_data():
    frames = []
    for wine_type, fname in FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"Downloading {fname} from UCI repository...")
            urllib.request.urlretrieve(BASE_URL + fname, path)
        df = pd.read_csv(path, sep=";")
        df["wine_type"] = 0 if wine_type == "red" else 1   # 0 = red, 1 = white
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


df = load_data()
print("Shape:", df.shape)
print(df.head())
print("\nMissing values per column:\n", df.isnull().sum())

# ------------------------------------------------------------- 2. PREPROCESS
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"\nRemoved {before - len(df)} duplicate rows -> {len(df)} rows remain")

df.describe().T.round(3).to_csv(f"{OUT_DIR}/dataset_summary.csv")

# Binary target for classification: Good wine = quality >= 6
df["good_quality"] = (df["quality"] >= 6).astype(int)

features = [c for c in df.columns if c not in ("quality", "good_quality")]
X = df[features]
y_reg = df["quality"]
y_clf = df["good_quality"]

# -------------------------------------------------------- 3. VISUALISATION (EDA)
plt.figure(figsize=(7, 4))
sns.countplot(x="quality", data=df, color="#1559a6")
plt.title("Distribution of Wine Quality Scores")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/01_quality_distribution.png", dpi=150); plt.close()

plt.figure(figsize=(11, 8))
sns.heatmap(df[features + ["quality"]].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", annot_kws={"size": 7})
plt.title("Correlation Heatmap")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/02_correlation_heatmap.png", dpi=150); plt.close()

plt.figure(figsize=(6, 4))
sns.boxplot(x="quality", y="alcohol", data=df, color="#93c5fd")
plt.title("Alcohol Content vs Quality")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/03_alcohol_vs_quality.png", dpi=150); plt.close()

# ------------------------------------------------------------ 4. TRAIN/TEST SPLIT
X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf)

# ------------------------------------------------------------- 5. REGRESSION
reg_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
}
reg_rows, reg_preds = [], {}
for name, model in reg_models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    pipe.fit(X_train, yr_train)
    pred = pipe.predict(X_test)
    reg_preds[name] = pred
    reg_rows.append({
        "Model": name,
        "MAE": mean_absolute_error(yr_test, pred),
        "RMSE": np.sqrt(mean_squared_error(yr_test, pred)),
        "R2": r2_score(yr_test, pred),
        "Rounded Accuracy": accuracy_score(yr_test, np.clip(np.rint(pred), 3, 9)),
    })
reg_df = pd.DataFrame(reg_rows).round(4).sort_values("R2", ascending=False)
reg_df.to_csv(f"{OUT_DIR}/regression_results.csv", index=False)
print("\n=== REGRESSION RESULTS ===\n", reg_df.to_string(index=False))

best_reg = reg_df.iloc[0]["Model"]
plt.figure(figsize=(5.5, 5))
plt.scatter(yr_test, reg_preds[best_reg], alpha=0.35, color="#1559a6")
plt.plot([3, 9], [3, 9], "r--")
plt.xlabel("Actual Quality"); plt.ylabel("Predicted Quality")
plt.title(f"Actual vs Predicted ({best_reg})")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/04_actual_vs_predicted.png", dpi=150); plt.close()

# -------------------------------------------------------- 6. CLASSIFICATION
clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "SVM (RBF)": SVC(kernel="rbf", C=2.0),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
}
clf_rows, fitted = [], {}
for name, model in clf_models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    pipe.fit(X_train, yc_train)
    pred = pipe.predict(X_test)
    fitted[name] = (pipe, pred)
    cv = cross_val_score(pipe, X, y_clf, cv=5, scoring="accuracy").mean()
    clf_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(yc_test, pred),
        "Precision": precision_score(yc_test, pred),
        "Recall": recall_score(yc_test, pred),
        "F1-Score": f1_score(yc_test, pred),
        "5-Fold CV Acc": cv,
    })
clf_df = pd.DataFrame(clf_rows).round(4).sort_values("Accuracy", ascending=False)
clf_df.to_csv(f"{OUT_DIR}/classification_results.csv", index=False)
print("\n=== CLASSIFICATION RESULTS ===\n", clf_df.to_string(index=False))

best_clf = clf_df.iloc[0]["Model"]
ConfusionMatrixDisplay(confusion_matrix(yc_test, fitted[best_clf][1]),
                       display_labels=["Not Good", "Good"]).plot(cmap="Blues")
plt.title(f"Confusion Matrix ({best_clf})")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/05_confusion_matrix.png", dpi=150); plt.close()

# -------------------------------------------------------- 7. FEATURE IMPORTANCE
rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, yc_train)
imp = pd.Series(rf.feature_importances_, index=features).sort_values()
plt.figure(figsize=(7, 5))
imp.plot(kind="barh", color="#1559a6")
plt.title("Feature Importance (Random Forest)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/06_feature_importance.png", dpi=150); plt.close()
imp.sort_values(ascending=False).round(4).to_csv(f"{OUT_DIR}/feature_importance.csv", header=["importance"])

print(f"\nBest regression model     : {best_reg}")
print(f"Best classification model : {best_clf}")
print(f"Top 3 features            : {list(imp.sort_values(ascending=False).index[:3])}")
print(f"All results saved in '{OUT_DIR}/'")
