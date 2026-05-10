# STEP 1: Import Libraries
# ----------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

print("=" * 55)
print("  IRIS FLOWER CLASSIFICATION PROJECT")
print("=" * 55)

# ----------------------------------------------------------
# STEP 2: Load the Dataset
# (No download needed — built into scikit-learn!)
# ----------------------------------------------------------
iris = load_iris()

X = iris.data          # Features: sepal/petal length & width
y = iris.target        # Labels: 0=setosa, 1=versicolor, 2=virginica
feature_names = iris.feature_names
target_names  = iris.target_names

print("\n[1] Dataset Loaded!")
print(f"    Total samples  : {X.shape[0]}")
print(f"    Features       : {X.shape[1]}")
print(f"    Species        : {list(target_names)}")

# ----------------------------------------------------------
# STEP 3: Explore the Data
# ----------------------------------------------------------
df = pd.DataFrame(X, columns=feature_names)
df["species"] = [target_names[i] for i in y]

print("\n[2] First 5 rows of the dataset:")
print(df.head().to_string(index=False))

print("\n[3] Basic statistics:")
print(df.describe().round(2).to_string())

print("\n[4] Samples per species:")
print(df["species"].value_counts().to_string())

# ----------------------------------------------------------
# STEP 4: Visualize the Data
# ----------------------------------------------------------
print("\n[5] Generating visualizations...")

# --- Plot 1: Pairplot (relationships between all features) ---
sns.pairplot(df, hue="species", palette="Set2", height=2.2)
plt.suptitle("Iris Dataset — Feature Pair Plot", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig("iris_pairplot.png", dpi=120, bbox_inches="tight")
plt.show()
print("    Saved: iris_pairplot.png")

# --- Plot 2: Box plot (feature distribution per species) ---
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
colors = ["#2ecc71", "#3498db", "#e74c3c"]
for ax, feat in zip(axes.flatten(), feature_names):
    data_per_species = [df[df["species"] == sp][feat] for sp in target_names]
    bp = ax.boxplot(data_per_species, patch_artist=True, labels=target_names)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(feat.title(), fontsize=11)
    ax.set_ylabel("cm")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.suptitle("Feature Distribution by Species", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("iris_boxplot.png", dpi=120, bbox_inches="tight")
plt.show()
print("    Saved: iris_boxplot.png")

# ----------------------------------------------------------
# STEP 5: Prepare Data for Training
# ----------------------------------------------------------
# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n[6] Train/Test Split:")
print(f"    Training samples : {len(X_train)}")
print(f"    Testing  samples : {len(X_test)}")

# ----------------------------------------------------------
# STEP 6: Train Two Models
# ----------------------------------------------------------

# --- Model A: K-Nearest Neighbors ---
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_acc  = accuracy_score(y_test, knn_pred)

# --- Model B: Decision Tree ---
dtree = DecisionTreeClassifier(max_depth=4, random_state=42)
dtree.fit(X_train, y_train)
dt_pred = dtree.predict(X_test)
dt_acc  = accuracy_score(y_test, dt_pred)

print(f"\n[7] Model Accuracy:")
print(f"    KNN (k=3)        : {knn_acc * 100:.2f}%")
print(f"    Decision Tree    : {dt_acc  * 100:.2f}%")

# ----------------------------------------------------------
# STEP 7: Classification Report (Detailed Metrics)
# ----------------------------------------------------------
print("\n[8] KNN — Detailed Report:")
print(classification_report(y_test, knn_pred, target_names=target_names))

print("[9] Decision Tree — Detailed Report:")
print(classification_report(y_test, dt_pred, target_names=target_names))

# ----------------------------------------------------------
# STEP 8: Confusion Matrix
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, preds, title in zip(
    axes,
    [knn_pred, dt_pred],
    ["KNN (k=3)", "Decision Tree"],
):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f"Confusion Matrix — {title}", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("iris_confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.show()
print("[10] Saved: iris_confusion_matrix.png")

# ----------------------------------------------------------
# STEP 9: Try KNN with Different k Values
# ----------------------------------------------------------
print("\n[11] Testing KNN with different k values:")
k_values  = range(1, 16)
k_scores  = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    score = accuracy_score(y_test, model.predict(X_test))
    k_scores.append(score)
    print(f"    k={k:2d}  →  {score * 100:.1f}%")

plt.figure(figsize=(8, 4))
plt.plot(k_values, [s * 100 for s in k_scores], marker="o", color="#3498db", linewidth=2)
plt.xlabel("k (number of neighbors)")
plt.ylabel("Accuracy (%)")
plt.title("KNN Accuracy vs k Value")
plt.xticks(k_values)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("iris_knn_k_values.png", dpi=120, bbox_inches="tight")
plt.show()
print("    Saved: iris_knn_k_values.png")

best_k = k_values[k_scores.index(max(k_scores))]
print(f"\n    Best k = {best_k}  (accuracy: {max(k_scores) * 100:.1f}%)")

# ----------------------------------------------------------
# STEP 10: Predict Your Own Flower!
# ----------------------------------------------------------
print("\n" + "=" * 55)
print("  PREDICT YOUR OWN FLOWER")
print("=" * 55)

# Change these values and re-run to see different predictions!
my_flower = [
    5.1,   # sepal length (cm)
    3.5,   # sepal width  (cm)
    1.4,   # petal length (cm)
    0.2,   # petal width  (cm)
]

prediction_idx  = knn.predict([my_flower])[0]
prediction_name = target_names[prediction_idx]

print(f"\n  Input measurements: {my_flower}")
print(f"  Predicted species : {prediction_name.upper()}")
print(f"\n  Species guide:")
print(f"    setosa    — very small petals (petal length < 2.5 cm)")
print(f"    versicolor — medium petals   (petal length 3–5 cm)")
print(f"    virginica  — large petals    (petal length > 5 cm)")

# ----------------------------------------------------------
print("\n" + "=" * 55)
print("  PROJECT COMPLETE! Check the saved .png files.")
print("=" * 55)
