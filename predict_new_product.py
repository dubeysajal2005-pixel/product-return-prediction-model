"""
Predict Product Return - using an ALREADY TRAINED model
------------------------------------------------------------
This script does NOT retrain anything. It loads the model saved
by train_decision_tree_cleaned.py (the file 'trained_model.pkl')
and lets you check products instantly.

IMPORTANT: Run train_decision_tree_cleaned.py at least ONCE first
(so that 'trained_model.pkl' gets created in this folder). After
that, you only ever need to run THIS script.

Run with:  python3 predict_new_product.py
"""

import os
import joblib
import pandas as pd

MODEL_FILE = "trained_model.pkl"

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"'{MODEL_FILE}' not found. You need to run "
        f"'train_decision_tree_cleaned.py' ONE TIME first - "
        f"it will create this file. After that, just run this script."
    )

# -----------------------------------------------------------
# Load the saved model + supporting data (instant, no training)
# -----------------------------------------------------------
bundle = joblib.load(MODEL_FILE)
model = bundle["model"]
location_risk = bundle["location_risk"]
overall_rate = bundle["overall_rate"]

print("Model loaded successfully (no retraining needed).")


def predict_return(price, rating, delivery_time, seller_rating, location):
    loc_risk = location_risk.get(location, overall_rate)
    new_data = pd.DataFrame([[
        price, rating, delivery_time, seller_rating, loc_risk
    ]], columns=["price", "rating", "delivery_time",
                 "defect_risk_proxy", "delivery_location_risk_proxy"])

    prediction = model.predict(new_data)[0]
    probability = model.predict_proba(new_data)[0]

    label = "Returned" if prediction == 1 else "Not Returned"
    print(f"\nNew Electronics product -> Price: {price}, Rating: {rating}, "
          f"Delivery: {delivery_time} days, Seller Rating: {seller_rating}, Location: {location}")
    print(f"Prediction: {label}")
    print(f"Probability -> Not Returned: {probability[0]:.2%}, Returned: {probability[1]:.2%}")
    return label, probability


def get_user_input_and_predict():
    print("\n" + "=" * 50)
    print("Naya Electronics product check karo - values daalo:")
    print("=" * 50)
    try:
        price = float(input("Price (e.g. 25000): "))
        rating = float(input("Rating (0 to 5, e.g. 4.2): "))
        delivery_time = int(input("Delivery time (days, e.g. 5): "))
        seller_rating = float(input("Seller rating (0 to 5, e.g. 4.0): "))
        location = input(f"Location {list(location_risk.index)}: ").strip()
    except ValueError:
        print("Galat input! Sirf numbers daalo price/rating/delivery_time/seller_rating ke liye.")
        return

    if location not in location_risk.index:
        print(f"Warning: '{location}' training data mein nahi mila, average risk use hoga.")

    predict_return(price, rating, delivery_time, seller_rating, location)


while True:
    get_user_input_and_predict()
    again = input("\nEk aur product check karna hai? (y/n): ").strip().lower()
    if again != "y":
        print("Dhanyawad!")
        break
