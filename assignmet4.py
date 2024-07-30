import pandas as pd
import numpy as np

#Import Data Set
df = pd.read_csv('Mental_illness.csv')
print(df)

# Display the first few rows
print(df.head())

# Display the last few rows
print(df.tail())

# Get the shape of the dataset (rows, columns)
print(df.shape)

# Display column names
print(df.columns)

# Get information about the dataset
print(df.info())

# Check for missing values
print(df.isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns

# Visualize missing values
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.show()

# Get summary statistics for numerical columns
print(df.describe())

# Get summary statistics for categorical columns
print(df.describe(include=['object', 'category']))

#Cleaning Dataset
#df =df.drop(['Entity', 'Code', 'Year'], axis=1)
df.to_csv('Mental_illness_update.csv')

#Data Prediction:
data= pd.read_csv('Mental_illness_update.csv')
X = data.drop(['Eating disorders '], axis=1)
Y = data['Eating disorders ']
Y=Y.values.reshape(-1, 1)

import sklearn as sk
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X, Y)

inp = np.array([[2],[0.17],[5.2], [3.5],[0.6]])
inp= inp.reshape(1, -1)

print('the preception in Eating Disorder for the input is:', clf.predict(inp))

day_index= 798
days = [i for i in range(Y.size)]

#Data Visualization:
print(' the preception trend Graph: ')
plt.scatter(days, Y, color='g')
plt.scatter(days[day_index], Y [day_index], color='r')
plt.title('Perception level')
plt.xlabel('Days')
plt.ylabel('Preceptation in inches')
plt.show()

x_f = X.filter(['Schizophrenia disorders ', 'Depressive disorders ', 'Bipolar disorders ',
                'Anxiety disorders ',], axis=1)
print('Preciptiation Vs Selected Attributes Graph: ')
for i in range(x_f.columns.size):
    plt.subplot(2, 2, i+1)
    plt.scatter(days, x_f[x_f.columns.values[i][:100]], color='g')
    plt.scatter(days[day_index], x_f[x_f.columns.values[i]]
                [day_index], color='r')
    plt.title(x_f.columns.values[i])
plt.show()
