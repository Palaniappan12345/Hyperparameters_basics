mport pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("diabetes.csv")
print(data.head())
print(data.describe())
print(data.head(20))
data = pd.read_csv("diabetes.csv",header=None)
print((data[[1,2,3,4,5]] == 0).sum())

# Mark zero values as missing or NaN
data[[1,2,3,4,5]] = data[[1,2,3,4,5]].replace(0, np.NaN)
# Count the number of NaN values in each column
print(data.isnull().sum())
#print(data.head())

# Fill missing values with mean column values
data.fillna(data.mean(), inplace=True)
# Count the number of NaN values in each column
print(data.isnull().sum())

# Split dataset into inputs and outputs
values = data.values
X = values[:,0:6]
y = values[:,6]
#print(X)
#print(Y)
lr = LogisticRegression(penalty='l2',dual=False,max_iter=110)

# Pass data to the LR model
lr.fit(X,y)

print(lr.score(X,y))
