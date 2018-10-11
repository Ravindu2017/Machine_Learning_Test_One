import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

#Data is gathered from the file destination and read through pandas
data = pd.read_csv("C:\\Users\\ravindur\\Desktop\\Resume\\Features_fixed.csv")
df = pd.DataFrame(data)
df = df.set_index('Date')
#Split the dataframe into two
df_2 = df[['Precipitation (mm)', 'Snow depth (cm)', 'Air Temperature (degC)', 'Maximum Temperature (degC)', 'Minimum Temperature (degC)']]
df_3 = df_2['Minimum Temperature (degC)']

#fill in the empty values
df_2.fillna(-99999, inplace=True)
#Then replace them so data is consistent(X and Y length)
df_2.dropna(inplace=True)
#X is everything but Minimum Temperature and Y is vice versa
X = np.array(df_2.drop(['Minimum Temperature (degC)'],1))
y = np.array(df_3)
#print(df.iloc[1])

#Data is split into training and test, 80/20 split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

#Model is opened from a pickle file
pickle_in = open("C:\\Users\\ravindur\\Desktop\\Resume\\Learning_Model.pickle", "rb")
machine = pickle.load(pickle_in)
#Score is measured using the model
#print(machine.score(X_test,y_test))
'''
#Now using the model to compare with real time values outside the original
#Data-set for new values
new_data = pd.read_csv("C:\\Users\\ravindur\\Desktop\\Resume\\New_Feautures.csv")
df_5 = pd.DataFrame(new_data)
df_5 = df_5.set_index('Date')
df_5.fillna(-99999, inplace=True)

df_5.dropna(inplace=True)
dff = df_5[['Precipitation (mm)', 'Snow depth (cm)', 'Air Temperature (degC)', 'Maximum Temperature (degC)', 'Minimum Temperature (degC)']]
#New X matrix
D = np.array(dff.drop(['Minimum Temperature (degC)'],1))
#print(D)
#for i in range(0,10
#New list just to validate if the model works
#S contains ten elements of X Matrix in MashinChurnin
S = []
for i in range(29,40):
    S.append(df[['Precipitation (mm)', 'Snow depth (cm)', 'Air Temperature (degC)', 'Maximum Temperature (degC)']].iloc[i])
#S.append(df[['Precipitation (mm)', 'Snow depth (cm)', 'Air Temperature (degC)', 'Maximum Temperature (degC)', 'Minimum Temperature (degC)']].iloc[0])
S = pd.DataFrame(S)
S.fillna(-99999, inplace=True)
S.dropna(inplace=True)
print(S)
#Predictions of y variable based on ten elements from X matrix
print(machine.predict(S))
'''

