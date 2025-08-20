import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sys
np.set_printoptions(threshold=sys.maxsize)
import pickle

#Importing DataSet 
dataset = pd.read_csv(r"C:\Users\DELL\Desktop\FSDS\ML\15th- SLR\SLR - House price prediction\House_data.csv")
space=dataset['sqft_living']
price=dataset['price']

x = np.array(space).reshape(-1, 1)
y = np.array(price)

#Splitting the data into Train and Test
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=1/3, random_state=0)

#Fitting simple linear regression to the Training Set
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

#Predicting the prices
pred = regressor.predict(xtest)

#Visualizing the training Test Results 
plt.scatter(xtrain, ytrain, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

#Visualizing the Test Results 
plt.scatter(xtest, ytest, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title("Visuals for Test DataSet")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

# Save the trained model to disk
filename = 'house_price_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)

# Load the pickle file
with open('house_price_model.pkl','rb') as file:
    loaded_model = pickle.load(file)

print("Model has been pickled and saved as house_price_model.pkl")

import os
print(os.getcwd())