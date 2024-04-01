# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
```

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Hemakesh G
RegisterNumber:  212223040064
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('/content/mldataset1.csv')
df.head(10)

plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')

x=df.iloc[:,0:1]
y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x_train,y_train)

x_train
y_train

lr.predict(x_test.iloc[0].values.reshape(1,1))

plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_train,lr.predict(x_train),color='blue')

lr.coef_
lr.intercept_
```

## Output:
## 1. Head:
![Screenshot 2024-04-01 154657](https://github.com/HEMAKESHG/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870552/b55e1147-dfbf-46eb-ad5a-b6fd7c5962ae)
## 2. Graph of plotted data:
![Screenshot 2024-04-01 154951](https://github.com/HEMAKESHG/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870552/d0851e17-e3b1-44bc-bcdd-a78cb5b16467)
## 3. Trained data:
![Screenshot 2024-04-01 155644](https://github.com/HEMAKESHG/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870552/67f3c32c-dd8a-4283-968f-0cc3e38a61a7)
## 4. Line of Regression:
![Screenshot 2024-04-01 155632](https://github.com/HEMAKESHG/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870552/e7fd8f20-7c82-4e86-86c6-0cafe6136b08)
![Screenshot 2024-04-01 155624](https://github.com/HEMAKESHG/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870552/c03e0991-49a9-42c6-9ac7-ae349ffcf083)
## 5. Coefficient and intercept values:
![Screenshot 2024-04-01 155502](https://github.com/HEMAKESHG/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870552/5a069bfe-c349-44e6-93c8-d26474b433d3)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
