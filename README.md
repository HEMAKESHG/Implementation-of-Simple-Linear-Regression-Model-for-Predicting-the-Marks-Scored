# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y.
2. Plot the independent variable X and dependent variable Y.
3. Predict the values.
4. Plot the predicted values.

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

```

## Output:
![image](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870552/36215328-94a9-4575-aef7-2e0a66d84883)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
