# Implementation of Univariate Linear Regression
## AIM:
To implement univariate Linear Regression to fit a straight line using least squares.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y.
2. Calculate the mean of the X -values and the mean of the Y -values.
3. Find the slope m of the line of best fit using the formula. 
<img width="231" alt="image" src="https://user-images.githubusercontent.com/93026020/192078527-b3b5ee3e-992f-46c4-865b-3b7ce4ac54ad.png">
4. Compute the y -intercept of the line by using the formula:
<img width="148" alt="image" src="https://user-images.githubusercontent.com/93026020/192078545-79d70b90-7e9d-4b85-9f8b-9d7548a4c5a4.png">
5. Use the slope m and the y -intercept to form the equation of the line.
6. Obtain the straight line equation Y=mX+b and plot the scatterplot.

## Program:
```
/*
Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: Preethika N

RegisterNumber:212223040130  
*/
```
Program to implement the linear regression using gradient descent.
Developed by: Vineela Shaik
RegisterNumber:  212223040243
*/
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01, num_iters=1000):
    x=np.c_[np.ones(len(x1)),x1]
    theta=np.zeros(x.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(x).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(x1))*x.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())
x=(data.iloc[1:,:-2].values)
print(x)
x1=x.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
x1_Scaled=scaler.fit_transform(x1)
y1_Scaled=scaler.fit_transform(y)
print(x1_Scaled)
print(y1_Scaled)
theta=linear_regression(x1_Scaled,y1_Scaled)
new_data=np.array([165349.2,136897.8,471781.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value: {pre}")
```
## Output:

![image](https://github.com/AkilaMohan/Find-the-best-fit-line-using-Least-Squares-Method/assets/155142246/c8e80c23-3545-4cd1-8c17-93db2b53a3e9)

![image](https://github.com/AkilaMohan/Find-the-best-fit-line-using-Least-Squares-Method/assets/155142246/9ecd7b68-d190-445e-b4e9-f9022bae7cf1)

## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
