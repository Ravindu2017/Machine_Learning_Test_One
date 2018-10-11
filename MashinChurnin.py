import pandas as pd
import math
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv("C:\\Users\\ravindur\\Desktop\\Resume\\Features_fixed.csv")
df = pd.DataFrame(data)
df = df.set_index('Date')
df_2 = df[['Precipitation (mm)', 'Snow depth (cm)', 'Air Temperature (degC)', 'Maximum Temperature (degC)', 'Minimum Temperature (degC)']]
df_3 = df_2['Minimum Temperature (degC)']

'''
forecast_col = 'Maximum Temperature (degC)'
df_2.fillna(-99999, inplace=True)
forecast_out = 30
print(forecast_out)
df_2['Label'] = df_2[forecast_col].shift(-forecast_out)
df_2.dropna(inplace=True)
X = np.array(df_2.drop(['Label'],1))
y = np.array(df_2['Label'])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
print (X_train.shape,y_train.shape)
print (X_test.shape,y_test.shape)
lm = linear_model.LinearRegression()
model = lm.fit(X_train,y_train)
prediction = lm.predict(X_test)
print (prediction[0:5])
you = model.score(X_test, y_test)
print(you)


'''
df_2.fillna(-99999, inplace=True)
#print(df_2)
#hhh
df_2.dropna(inplace=True)
X = np.array(df_2.drop(['Minimum Temperature (degC)'],1))
y = np.array(df_3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
print (X_train.shape,y_train.shape)
print (X_test.shape,y_test.shape)

#lm = linear_model.LinearRegression()
##model = lm.fit(X_train,y_train)
##with open("C:\\Users\\ravindur\\Desktop\\Resume\\Learning_Model.pickle", "wb") as f:
##    pickle.dump(model, f)

pickle_in = open("C:\\Users\\ravindur\\Desktop\\Resume\\Learning_Model.pickle", "rb")
model = pickle.load(pickle_in)

#prediction = lm.predict(X_test)
#print (prediction[0:5])
you = model.score(X_test, y_test)
print(you)

##plt.scatter(y_test, prediction)
##plt.xlabel("True Values")
##plt.ylabel("Prediction!")
##plt.show()
#print(

'''
passing floats is a no no
we need integers

#Need to drop the date time objects because its messing up my scriptoid
'''
