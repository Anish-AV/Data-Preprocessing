#importing libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

#importing data set

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#handling missing values
imputer= SimpleImputer(missing_values=np.nan, strategy='mean')
imputer=imputer.fit(x[:,1:3])
x[:,1:3]= imputer.transform(x[:,1:3])
