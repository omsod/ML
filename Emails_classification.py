import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load dataset
df = pd.read_csv("emails.csv")
df.shape

# Step 2: Separate features and labels
X = df.drop(columns=['Email No.', 'Prediction'], errors='ignore')
y = df['Prediction']

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train models
# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# SVM
svm = SVC(kernel='linear', C=1)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# Step 5: Evaluate
def evaluate(y_true, y_pred, model_name):
    print(f"\n🔹 {model_name} Performance:")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print(classification_report(y_true, y_pred, target_names=['Not Spam', 'Spam']))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

evaluate(y_test, y_pred_knn, "KNN")
evaluate(y_test, y_pred_svm, "SVM")

# Step 6: Compare
acc_knn = accuracy_score(y_test, y_pred_knn)
acc_svm = accuracy_score(y_test, y_pred_svm)

print("\n Model Comparison:")
print(f"KNN Accuracy: {acc_knn:.4f}")
print(f"SVM Accuracy: {acc_svm:.4f}")

if acc_svm > acc_knn:
    print(" SVM performs better on this dataset.")
elif acc_knn > acc_svm:
    print(" KNN performs better on this dataset.")
else:
    print(" Both models performed equally well.")
