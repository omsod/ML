import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Step 1: Load dataset
df = pd.read_csv("diabetes.csv")
print(" Dataset loaded successfully!")
print(df.shape)

# Step 2: Split into features and target
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Normalize data (important for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Find best k-value
error_rates = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error = np.mean(y_pred != y_test)
    error_rates.append(error)

best_k = np.argmin(error_rates) + 1
print(f"\n Best K found: {best_k}")

# Step 6: Train final KNN model with best_k
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


# Step 7: Evaluate metrics
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)


# Step 8: Display results
print("\n Final Model Evaluation Results:")
print("Confusion Matrix:\n", cm)
print("Accuracy:", round(accuracy, 4))
print("Error Rate:", round(error_rate, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))


import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.plot(range(1, 21), [1 - e for e in error_rates], marker='o')
plt.title("KNN Accuracy vs K Value")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
