#importing libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing data 

dataset = pd.read_csv('C:/Users/My Pc/Desktop/machine learning tests/regression/Polynomial Regression/Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values 
y = dataset.iloc[:, 2].values


# Fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg1 = LinearRegression()
lin_reg1.fit(x, y)


# Fitting polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures 
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)


# visualising linear regression
plt.scatter(x, y, c = 'r' )
plt.plot(x, lin_reg1.predict(x), c = 'y')
plt.title('salary vs possiotion (polynomial regression)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()


# visualising polynomial regression
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, c = 'g')
plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)), c = 'r')
plt.title('salary vs possiotion (polynomial regression)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()


# predicitng a new result with linear regression
lin_reg1.predict(np.array([6.5]).reshape(1, 1))


# predicitng a new result with polynomial regresion
lin_reg2.predict(poly_reg.fit_transform(np.array([6.5]).reshape(1, 1)))
