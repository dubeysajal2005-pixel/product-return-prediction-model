"""
E-commerce Product Return Prediction - RULE-BASED MODEL
(NO ML classifier used - no DecisionTreeClassifier, no
 RandomForestClassifier, no sklearn model training at all)
----------------------------------------------------------------
Instead of an algorithm LEARNING patterns from data, WE manually
write if-else rules based on patterns found during EDA:

  1. delivery_time == 6 days  -> return rate jumps to ~20.6%
     (vs ~10% for delivery_time 1-5 days)
  2. rating <= 3              -> return rate is ~19.2%
     (vs ~11% for rating > 3)

RULE: If EITHER condition is true, predict "Returned".
      Otherwise, predict "Not Returned".

This is called a RULE-BASED / HEURISTIC model - a common baseline
that data scientists build BEFORE trying ML models, to check
whether the extra complexity of ML is actually worth it.

Run with:  python3 rule_based_model.py

Works in Google Colab too - if the CSV isn't found, it will
prompt a file-upload dialog automatically.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
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

# -----------------------------------------------------------
# STEP 1: Load data
# -----------------------------------------------------------
df = pd.read_csv(FILE_NAME)
target = "is_returned"
print("Dataset shape:", df.shape)

# -----------------------------------------------------------
# STEP 2: Same train/test split as the ML models (for fair comparison)
# -----------------------------------------------------------
df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df[target]
)

# -----------------------------------------------------------
# STEP 3: Explore return-rate patterns on TRAINING data ONLY
#   (this is how we DECIDE the rule thresholds - manually,
#    by looking at the numbers, not by training a model)
# -----------------------------------------------------------
print("\nReturn rate by delivery_time (on training data):")
print(df_train.groupby("shipping_time_days")[target].mean())

print("\nReturn rate by rating bucket (on training data):")
rating_buckets = pd.cut(df_train["rating"], bins=[0, 2, 3, 4, 5])
print(df_train.groupby(rating_buckets, observed=True)[target].mean())

# -----------------------------------------------------------
# STEP 4: Define the RULE (manual, no learning algorithm)
#   Based on the patterns above:
#     - delivery_time == 6 has a noticeably higher return rate
#     - rating <= 3 has a noticeably higher return rate
# -----------------------------------------------------------
def rule_based_predict(row):
    if row["shipping_time_days"] == 6:
        return 1  # Returned
    if row["rating"] <= 3:
        return 1  # Returned
    return 0  # Not Returned

# -----------------------------------------------------------
# STEP 5: Apply the rule to the TEST set (no .fit(), no training)
# -----------------------------------------------------------
y_test = df_test[target].astype(int)
y_pred = df_test.apply(rule_based_predict, axis=1)

# -----------------------------------------------------------
# STEP 6: Evaluate - same metrics as the ML models, for comparison
# -----------------------------------------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n===== RULE-BASED MODEL EVALUATION (no classifier used) =====")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nFull classification report:")
print(classification_report(y_test, y_pred, target_names=["Not Returned", "Returned"]))

print("\n===== COMPARISON WITH ML MODELS (from earlier) =====")
print("Decision Tree : Accuracy=0.8291, Precision=0.2061, Recall=0.1862, F1=0.1957")
print("Random Forest : Accuracy=0.8295, Precision=0.2067, Recall=0.1858, F1=0.1957")
print(f"Rule-Based    : Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

# =============================================================
# STEP 7: Predict for a NEW product (manual rule, instant)
# =============================================================
def predict_return_rule_based(delivery_time, rating):
    if delivery_time == 6 or rating <= 3:
        label = "Returned"
    else:
        label = "Not Returned"
    print(f"\nDelivery time: {delivery_time} days, Rating: {rating} -> Prediction: {label}")
    return label


def get_user_input_and_predict():
    print("\n" + "=" * 50)
    print("Naya Electronics product check karo (RULE-BASED):")
    print("=" * 50)
    try:
        delivery_time = int(input("Delivery time (days, 1-6): "))
        rating = float(input("Rating (0 to 5): "))
    except ValueError:
        print("Galat input! Sirf numbers daalo.")
        return
    predict_return_rule_based(delivery_time, rating)


while True:
    get_user_input_and_predict()
    again = input("\nEk aur product check karna hai? (y/n): ").strip().lower()
    if again != "y":
        print("Dhanyawad!")
        break