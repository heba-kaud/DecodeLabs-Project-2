# 📊 Order Status Prediction using KNN

## 📌 Project Overview

This project focuses on predicting the status of an order using machine learning.

The dataset contains information about customer orders, including product, quantity, unit price, payment method, items in cart, referral source, and total price.

The target variable is `OrderStatus`.

## 🎯 Objective

The main objective is to build a classification model that predicts the order status.

The possible order statuses are:

- Cancelled
- Delivered
- Pending
- Returned
- Shipped

## 📂 Dataset

The dataset used in this project is:

`Dataset for Data Analytics.csv`

It contains 14 columns and includes information such as:

- OrderID
- Date
- CustomerID
- Product
- Quantity
- UnitPrice
- ShippingAddress
- PaymentMethod
- OrderStatus
- TrackingNumber
- ItemsInCart
- CouponCode
- ReferralSource
- TotalPrice

## 🧩 Features

The selected features used for prediction are:

- Product
- Quantity
- UnitPrice
- PaymentMethod
- ItemsInCart
- ReferralSource
- TotalPrice

Categorical features were converted into numerical representations using One-Hot Encoding.

Numerical features were scaled before training the model.

## 🎯 Target

The target variable is:

`OrderStatus`

## 🤖 Machine Learning Model

The classification model used in this project is:

**K-Nearest Neighbors (KNN)**

KNN was used to classify orders into their corresponding order status categories.

## ⚙️ Data Processing

The project follows these main steps:

1. Load the dataset using Pandas.
2. Select the features and target.
3. Identify categorical and numerical features.
4. Apply One-Hot Encoding to categorical features.
5. Split the data into training and testing sets.
6. Scale the features using StandardScaler.
7. Train the KNN classifier.
8. Make predictions on the test data.
9. Evaluate the model.

## 📈 Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### Accuracy

The obtained accuracy was approximately:

**15.83%**

The classification report and confusion matrix are also included in the project output.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn

## ▶️ How to Run

1. Install Python.
2. Install the required libraries.
3. Open the project in PyCharm or another Python IDE.
4. Make sure the dataset is in the same project directory.
5. Run:
