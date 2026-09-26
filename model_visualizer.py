import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Sample E-commerce Dataset create karna (Price aur Rating features)
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Teeno Models Train Karna
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
rf_model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42).fit(X_train, y_train)
lr_model = LogisticRegression().fit(X_train, y_train)

# 3. Accuracy Calculate karna
dt_acc = accuracy_score(y_test, dt_model.predict(X_test))
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
lr_acc = accuracy_score(y_test, lr_model.predict(X_test))

# -------------------------------------------------------------
# NAYA EDIT: CMD Terminal par percentages print karne ke liye
# -------------------------------------------------------------
print("\n" + "="*35)
print("     Model Evaluation Results    ")
print("="*35)
print(f"Decision Tree Accuracy:    {dt_acc * 100:.2f}%")
print(f"Random Forest Accuracy:    {rf_acc * 100:.2f}%")
print(f"Logistic Regression Accuracy: {lr_acc * 100:.2f}%")
print("="*35 + "\n")
# -------------------------------------------------------------

# 4. Ek Hi Canvas Par 3 Plots Draw Karna
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Decision Tree Tree Structure
plot_tree(dt_model, feature_names=['Price', 'Rating'], class_names=['Not Returned', 'Returned'], filled=True, ax=axes[0], fontsize=8)
axes[0].set_title("1. Decision Tree Visualization", fontsize=12, fontweight='bold')

# Plot 2: Logistic Regression Sigmoid Probability Curve
x_vals = np.linspace(-6, 6, 100)
y_vals = 1 / (1 + np.exp(-x_vals))
axes[1].plot(x_vals, y_vals, color='green', linewidth=2.5, label='Sigmoid Curve')
axes[1].axhline(0.5, color='red', linestyle='--', label='Decision Threshold (0.5)')
axes[1].set_title("2. Logistic Regression Curve", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Linear Feature Combination")
axes[1].set_ylabel("Probability (0 to 1)")
axes[1].legend()
axes[1].grid(True)

# Plot 3: Teeno Models Ki Accuracy Comparison Chart
models = ['Decision Tree', 'Random Forest', 'Logistic Reg.']
accuracies = [dt_acc, rf_acc, lr_acc]
bars = axes[2].bar(models, accuracies, color=['#3498db', '#e67e22', '#2ecc71'])
axes[2].set_ylim(0, 1.15)
axes[2].set_title("3. Models Accuracy Comparison", fontsize=12, fontweight='bold')
axes[2].set_ylabel("Accuracy Score")

# Bars par values likhna
for bar in bars:
    yval = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2.0, yval + 0.03, f"{yval:.2f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()