# sklearn_mlp_churn.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load
df = pd.read_csv("Churn_Modelling.csv")

# Defensive: check columns
expected = ['RowNumber','CustomerId','Surname','CreditScore','Geography','Gender',
            'Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited']
missing = [c for c in expected if c not in df.columns]
if missing:
    print("Warning: these expected columns are missing:", missing)

# Features and target
X = df.copy()
for drop_col in ['RowNumber','CustomerId','Surname']:
    if drop_col in X.columns:
        X = X.drop(columns=drop_col)
y = X.pop('Exited')  # remove target from X, keep y

# One-hot encode categorical features safely
X = pd.get_dummies(X, columns=['Geography','Gender'], drop_first=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()>1 else None
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build & train MLP
mlp = MLPClassifier(hidden_layer_sizes=(32,16), activation='relu', solver='adam',
                    max_iter=200, random_state=42)
mlp.fit(X_train, y_train)

# Predict & evaluate
y_pred = mlp.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred),4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
