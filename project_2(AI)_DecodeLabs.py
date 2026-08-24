import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#Load Dataset, Define features and target and Define features type
df = pd.read_csv("Dataset for Data Analytics.csv")
print(df.head())
print(df.columns)
print(df["OrderStatus"].unique())
features = ["Product","Quantity","UnitPrice","PaymentMethod","ItemsInCart","ReferralSource","TotalPrice"]
target = "OrderStatus"
X = df[features]
y = df[target]
print(y.value_counts())
categorical_features = ["Product","PaymentMethod","ReferralSource"]
numeric_features = ["Quantity","UnitPrice","ItemsInCart","TotalPrice"]
print(pd.crosstab(df["PaymentMethod"],df["OrderStatus"]))
print(pd.crosstab(df["Product"],df["OrderStatus"]))
print(pd.crosstab(df["ReferralSource"],df["OrderStatus"]))
print(df.head(10).to_string())



#train/test split part (80% training and 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\nTraining samples:",len(X_train))
print("Testing samples:",len(X_test))


#preprocessing part
preprocessor = ColumnTransformer(transformers=[("categorical",OneHotEncoder(handle_unknown="ignore"), categorical_features),("numeric", StandardScaler(), numeric_features)])


#Training part using KNN algorism when k = 5
model = Pipeline(steps=[("preprocessor",preprocessor), ("classifier", KNeighborsClassifier(n_neighbors=5))])
model.fit(X_train, y_train)
print("\nModel training completed successfully.")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy*100:.2f}%")

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))




