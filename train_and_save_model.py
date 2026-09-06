"""
E-commerce Product Return Prediction using Decision Tree
(Version 4: trained on the CLEANED Electronics dataset -
 outliers already removed via IQR method)
----------------------------------------------------------------
Features used: price, rating, delivery_time,
               defect_risk_proxy (seller_rating),
               delivery_location_risk_proxy (location_return_rate)
----------------------------------------------------------------
Run with:  python3 train_decision_tree_cleaned.py

Works in Google Colab too - if the CSV isn't found, it will
prompt a file-upload dialog automatically.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)

FILE_NAME = "amazon_ecommerce_electronics_CLEANED.csv"

# -----------------------------------------------------------
# STEP 0: Auto-upload if file not found (Colab)
# -----------------------------------------------------------
if not os.path.exists(FILE_NAME):
    try:
        from google.colab import files
        print(f"'{FILE_NAME}' not found - please choose the file to upload:")
        uploaded = files.upload()
        FILE_NAME = list(uploaded.keys())[0]
    except ImportError:
        raise FileNotFoundError(
            f"'{FILE_NAME}' not found. Place the CSV next to this script, "
            f"or run this in Google Colab so it can prompt an upload."
        )

os.makedirs("outputs", exist_ok=True)

# -----------------------------------------------------------
# STEP 1: Load the cleaned dataset (already Electronics-only,
#          duplicates removed, rating outliers removed)
# -----------------------------------------------------------
df = pd.read_csv(FILE_NAME)
print("Dataset shape:", df.shape)

target = "is_returned"

# -----------------------------------------------------------
# STEP 2: Train/test split FIRST (before location_return_rate,
#   to avoid leaking test-set info into training)
# -----------------------------------------------------------
df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df[target]
)

# -----------------------------------------------------------
# STEP 3: Engineer location_return_rate (delivery-location risk)
#   Computed ONLY on training data.
# -----------------------------------------------------------
location_risk = df_train.groupby("location")[target].mean()
overall_rate = df_train[target].mean()

df_train = df_train.copy()
df_test = df_test.copy()
df_train["location_return_rate"] = df_train["location"].map(location_risk)
df_test["location_return_rate"] = df_test["location"].map(location_risk).fillna(overall_rate)

print("\nDelivery-location risk (historical return rate by city, TRAIN data):")
print(location_risk.sort_values(ascending=False))

# -----------------------------------------------------------
# STEP 4: Build final feature set (NO category)
# -----------------------------------------------------------
feature_cols = {
    "price": "price",
    "rating": "rating",
    "shipping_time_days": "delivery_time",
    "seller_rating": "defect_risk_proxy",
    "location_return_rate": "delivery_location_risk_proxy"
}

X_train = df_train[list(feature_cols.keys())].rename(columns=feature_cols)
X_test = df_test[list(feature_cols.keys())].rename(columns=feature_cols)
y_train = df_train[target].astype(int)
y_test = df_test[target].astype(int)

print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print("Final feature columns:", list(X_train.columns))

# -----------------------------------------------------------
# STEP 5: Train the Decision Tree
# -----------------------------------------------------------
model = DecisionTreeClassifier(
    max_depth=5,
    criterion="gini",
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# -----------------------------------------------------------
# STEP 6: Predict + Evaluate
# -----------------------------------------------------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nFull classification report:")
print(classification_report(y_test, y_pred, target_names=["Not Returned", "Returned"]))

print("\nFeature importance:")
for feat, imp in zip(X_train.columns, model.feature_importances_):
    print(f"  {feat}: {imp:.4f}")

# -----------------------------------------------------------
# STEP 7: Visualize Confusion Matrix + Tree
# -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Returned", "Returned"])
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix - Decision Tree (Cleaned Data)")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix_cleaned.png", dpi=150)
plt.close()

plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=X_train.columns,
    class_names=["Not Returned", "Returned"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Decision Tree - Electronics Return Prediction (Cleaned Data)")
plt.tight_layout()
plt.savefig("outputs/decision_tree_cleaned.png", dpi=150)
plt.close()

print("\nSaved plots to outputs/confusion_matrix_cleaned.png and outputs/decision_tree_cleaned.png")

# -----------------------------------------------------------
# STEP 7b: Box Plot - Price distribution by Return status
# -----------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.boxplot(x="is_returned", y="price", data=df, hue="is_returned",
            palette=["#4C72B0", "#DD8452"], legend=False)
plt.title("Box Plot: Price Distribution by Return Status")
plt.xlabel("Is Returned")
plt.ylabel("Price (INR)")
plt.tight_layout()
plt.savefig("outputs/boxplot_price_return.png", dpi=150)
plt.close()
print("Saved plot to outputs/boxplot_price_return.png")

# -----------------------------------------------------------
# STEP 7c: Scatter Plot - Price vs Delivery Time
# -----------------------------------------------------------
plt.figure(figsize=(7, 5))
sample = df.sample(min(3000, len(df)), random_state=42)
colors = sample["is_returned"].map({True: "#DD8452", False: "#4C72B0"})
plt.scatter(sample["shipping_time_days"], sample["price"], c=colors, alpha=0.5, s=15)
plt.title("Scatter Plot: Price vs Delivery Time (sample)")
plt.xlabel("Delivery Time (days)")
plt.ylabel("Price (INR)")
handles = [mpatches.Patch(color="#4C72B0", label="Not Returned"),
           mpatches.Patch(color="#DD8452", label="Returned")]
plt.legend(handles=handles)
plt.tight_layout()
plt.savefig("outputs/scatter_price_delivery.png", dpi=150)
plt.close()
print("Saved plot to outputs/scatter_price_delivery.png")

# -----------------------------------------------------------
# STEP 7d: Correlation Heatmap
# -----------------------------------------------------------
num_cols = ["price", "rating", "shipping_time_days", "seller_rating"]
corr_df = df[num_cols + [target]].copy()
corr_df[target] = corr_df[target].astype(int)
corr_df = corr_df.rename(columns={"shipping_time_days": "delivery_time", target: "is_returned"})

plt.figure(figsize=(7, 6))
sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Heatmap (numeric features)")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150)
plt.close()
print("Saved plot to outputs/correlation_heatmap.png")

# -----------------------------------------------------------
# STEP 7e: Histogram / Density Plot
# -----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df["price"], kde=True, bins=40, ax=axes[0], color="#4C72B0")
axes[0].set_title("Histogram + Density: Price")
sns.histplot(df["shipping_time_days"], kde=True, bins=20, ax=axes[1], color="#55A868")
axes[1].set_title("Histogram + Density: Delivery Time")
plt.tight_layout()
plt.savefig("outputs/histogram_density.png", dpi=150)
plt.close()
print("Saved plot to outputs/histogram_density.png")

# -----------------------------------------------------------
# STEP 7f: Bar Chart - Return Rate (%) by Location
# -----------------------------------------------------------
plt.figure(figsize=(7, 5))
return_rate_by_loc = (df.groupby("location")[target].mean().sort_values(ascending=False) * 100)
sns.barplot(x=return_rate_by_loc.index, y=return_rate_by_loc.values,
            hue=return_rate_by_loc.index, palette="viridis", legend=False)
plt.title("Bar Chart: Return Rate (%) by Location")
plt.xlabel("Location")
plt.ylabel("Return Rate (%)")
plt.tight_layout()
plt.savefig("outputs/barchart_location_return.png", dpi=150)
plt.close()
print("Saved plot to outputs/barchart_location_return.png")

print("\nAll 5 EDA plots + Decision Tree + Confusion Matrix saved in 'outputs' folder.")

# =============================================================
# STEP 8: SAVE the trained model + supporting data to disk
#   This is the key step - once this file is saved, you never
#   need to retrain. Just run predict_new_product.py next time.
# =============================================================
model_bundle = {
    "model": model,
    "location_risk": location_risk,
    "overall_rate": overall_rate
}
joblib.dump(model_bundle, "trained_model.pkl")
print("\nModel saved to 'trained_model.pkl'")
print("Next time, just run: python predict_new_product.py")
print("(No need to retrain - it loads this saved file instantly.)")
