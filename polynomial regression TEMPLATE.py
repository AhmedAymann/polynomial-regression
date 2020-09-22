#importing libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing data 

dataset = pd.read_csv('C:/Users/My Pc/Desktop/machine learning tests/regression/Polynomial Regression/Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values 
y = dataset.iloc[:, 2].values

""" # splitting data into trainingset and testset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( dp, indp, test_size= 0.2, random_state= 0)"""


# Fitting the Regression to the dataset
#create your regressor here

# predicitng a new result with polynomial regresion
y_pred = regresor.predict(np.array([6.5]).reshape(1, 1))

# visualising polynomial regression
plt.scatter(x, y, c = 'g')
plt.plot(x_grid, regresor.predict(x_grid), c = 'r')
plt.title('salary vs possiotion (polynomial regression)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()


# visualising polynomial regression
x_grid = np.arrange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(), 1))
plt.scatter(x, y, c = 'g')
plt.plot(x_grid, regressor.predict(x_grid), c = 'r')
plt.title('salary vs possiotion (polynomial regression)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()