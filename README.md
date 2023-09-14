# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1 Upload the file to your compiler.
2.Type the required program.
3.Print the program.
4.End the program.. 
 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: arun.j
RegisterNumber:  212222040015
*/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1.txt", header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Popuation of city (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  m=len(y)
  h=X.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
  
data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(X,y,theta))
  return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0]for y in x_value]
plt.plot(x_value,y_value,color="purple")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions = np.dot(theta.transpose(),x)
    return predictions[0]
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000 , we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000 , we predict a profit of $"+str(round(predict2,0)))


```

## Output:
![Screenshot 2023-09-14 083652](https://github.com/arun1111j/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128461833/578f8c8e-5ac8-48ac-a2b5-3ca8793fbaa5)
![Screenshot 2023-09-14 083725](https://github.com/arun1111j/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128461833/6ab4660c-cc5a-492c-aa20-1c59f65f0a99)
![image](https://github.com/arun1111j/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128461833/e7fe26b6-6b8e-4a19-8bd7-b26b19200ce5)
![image](https://github.com/arun1111j/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128461833/d6b40a4f-3d4f-4140-9df6-1875df9a835f)
![Screenshot 2023-09-14 083913](https://github.com/arun1111j/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128461833/715fc47f-7b3f-471d-a56c-dcc2058eb18d)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
