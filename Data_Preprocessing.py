#importing libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


#importing data set

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#handling missing values
imputer= SimpleImputer(missing_values=np.nan, strategy='mean')
imputer=imputer.fit(x[:,1:3])
x[:,1:3]= imputer.transform(x[:,1:3])

#encoding data
le_x= LabelEncoder()
x[:,0]=le_x.fit_transform(x[:,0])

ohe_x=OneHotEncoder()
x = ohe_x.fit_transform(x).toarray()
le_y= LabelEncoder()
y=le_y.fit_transform(y)


#spliting dataset
x_train,  x_test,y_train, y_test= train_test_split(x,y, test_size= 0.2, random_state=0 )

#feature scaling
sc_x= StandardScaler()
x_train= sc_x.fit_transform(x_train)
x_test= sc_x.transform(x_test)