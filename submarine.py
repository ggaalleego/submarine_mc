#%%
#DEPENDENCIES
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#%%
#DATA PROCESSING & COLLECTING
#loading the dataset to a pandas dataframe
sonar_data = pd.read_csv('sonar_data.csv', header = None)
sonar_data.head()
#%%
#number of rows
sonar_data.shape
sonar_data.describe() #dscribe --> statistical measures of the data
#%%
sonar_data[60].value_counts()
# %%
sonar_data.groupby(60).mean()
#%%
#separating data and labels
x= sonar_data.drop(columns= 60, axis= 1)
y = sonar_data[60]
print(x)
print(y)
# %%
#Training and testing data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.1, stratify = y, random_state=1)
print(x.shape, x_train.shape, x_test.shape)
# %%
#Model training --> Logistic Regression
model = LogisticRegression()
model.fit(x_train, y_train)
# %%
#Model evaluation 
#accuracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print('Accuracy on training data: ', training_data_accuracy)
# %%
#accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print('Accuracy on training data: ', test_data_accuracy)
# %%
#Making a predicitve system either Rock or Mine
input_data = (0.0317,0.0956,0.1321,0.1408,0.1674,0.1710,0.0731,0.1401,0.2083,0.3513,0.1786,0.0658,0.0513,0.3752,0.5419,0.5440,0.5150,0.4262,0.2024,0.4233,0.7723,0.9735,0.9390,0.5559,0.5268,0.6826,0.5713,0.5429,0.2177,0.2149,0.5811,0.6323,0.2965,0.1873,0.2969,0.5163,0.6153,0.4283,0.5479,0.6133,0.5017,0.2377,0.1957,0.1749,0.1304,0.0597,0.1124,0.1047,0.0507,0.0159,0.0195,0.0201,0.0248,0.0131,0.0070,0.0138,0.0092,0.0143,0.0036,0.0103)
#changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)
#reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)
# %%
