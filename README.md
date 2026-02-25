# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset, separate features and target, scale the input features, and encode the target labels.
2.Split the dataset into training and testing sets using stratified sampling.

3.Train the Logistic Regression model using training data and predict the test data.

4.Evaluate the model using accuracy, confusion matrix, and classification report, then visualize the confusion matrix.
## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('food_items (1).csv')
print('Name:RAGHUL.S')
print('Reg. No: 212225040325')
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
X_raw = df.iloc[:,:-1]
y_raw = df.iloc[:,-1:]
scaler= MinMaxScaler()
X= scaler.fit_transform(X_raw)
label_encoder = LabelEncoder()
y= label_encoder.fit_transform(y_raw.values.ravel())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify =y, random_state=123)

penalty='l2'

multi_class='multinomial'

solver= 'lbfgs'

max_iter=1000

l2_model = LogisticRegression(
    random_state=123,
    penalty=penalty,
    multi_class=multi_class, 
    solver=solver, 
    max_iter=max_iter
)
l2_model.fit(X_train, y_train)

y_pred= l2_model.predict(X_test)
print('Name: RAGHUL.S')
print('Reg. No: 212225040325')
print("\nModel Evaluation:")
print("Accuracy:",accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

conf_matrix= confusion_matrix(y_test, y_pred)
print(conf_matrix)

print('Name: RAGHUL.S')
print('Reg. No: 212225040325')
/*
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: RAGHUL.S
RegisterNumber: 212225040325 
*/
```

## Output:
<img width="1473" height="649" alt="Screenshot 2026-02-25 111656" src="https://github.com/user-attachments/assets/6638e4e7-8787-4cd1-bf9c-a9b0407e277c" />

<img width="1190" height="611" alt="Screenshot 2026-02-25 111706" src="https://github.com/user-attachments/assets/53e5209d-9123-46ed-83ed-9feb99805a12" />

<img width="1204" height="368" alt="Screenshot 2026-02-25 111720" src="https://github.com/user-attachments/assets/8698741d-9950-4f54-9e6f-d7d1e9c2a4a5" />


<img width="1118" height="545" alt="Screenshot 2026-02-25 111728" src="https://github.com/user-attachments/assets/6ca234da-4f98-426e-a1cc-1756ece97812" />

## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
