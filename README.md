# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VARSHITHA A T
RegisterNumber: 212221040176 
*/
```
```
import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
print("Placement data")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
print("Salary data")
data1.head()

print("Checking the null() function")
data1.isnull().sum()

print("Data Duplicate")
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
print("Print data")
data1

x=data1.iloc[:,:-1]
print("Data-status")
x

y=data1["status"]
print("data-status")
y


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print("y_prediction array")
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy value")
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print("Confusion array")
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("Classification report")
print(classification_report1)

print("Prediction of LR")
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![image](https://github.com/varshithathirumalachari/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131793193/a68c765f-666f-48d7-9838-0f1232b935a7)
![image](https://github.com/varshithathirumalachari/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131793193/ff867d8e-f688-41b5-92d3-04630b74b74d)
![image](https://github.com/varshithathirumalachari/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131793193/8b111694-4822-4ee5-b18c-f65f943108c6)
![image](https://github.com/varshithathirumalachari/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131793193/4270a8c0-d3d4-4770-b5eb-c60c081e9a97)
![image](https://github.com/varshithathirumalachari/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131793193/40ffc2b7-e7d2-4809-bcca-587a69618069)
![image](https://github.com/varshithathirumalachari/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131793193/ab471269-ea90-4b72-930f-bcc4a5146c85)
![image](https://github.com/varshithathirumalachari/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131793193/320de862-6c69-43d5-9c53-a65a0e5139e0)
![image](https://github.com/varshithathirumalachari/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131793193/d259fab5-007e-4845-85d9-60d4dcb7d994)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
