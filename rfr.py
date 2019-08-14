# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:04:15 2019

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import dataset
dataset=pd.read_csv('Position_Salaries.Csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values
x_pred=np.array([[6.5]])
#data splitting
"""
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)
"""
#Fitting  regression model to dataset
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(X,y)
#Predicting polynomial regression
y_pred=regressor.predict(x_pred)

#visualization
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Random Forest regression')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()