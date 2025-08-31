import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\DELL\Desktop\FSDS\ML\emp_sal.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# decission tree 
from sklearn.tree import DecisionTreeRegressor

dtr_reg_model = DecisionTreeRegressor(criterion='absolute_error', max_depth=10, splitter='random')
dtr_reg_model.fit(X,y)

dtr_reg_pred = dtr_reg_model.predict([[6.5]])
print(dtr_reg_pred)


# Random forest 
from sklearn.ensemble import RandomForestRegressor

rfr_reg_model = RandomForestRegressor(n_estimators=8,random_state=0)
rfr_reg_model.fit(X,y)

rfr_reg_pred = rfr_reg_model.predict([[6.5]])
print(rfr_reg_pred)


