import pandas as pd
import numpy as numpy
import sklearn.preprocessing as preprocessing
from sklearn import model_selection
from sklearn.linear_model import LinearRegression

df = pd.read_csv("IRIS.csv")
df.species = pd.factorize(df.species)[0]
df.dropna(inplace = True)

x = df[['sepal_length','sepal_width','petal_length','petal_width']]
y = df[['species']]

x = preprocessing.scale(x)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size=0.2)

model = LinearRegression()
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

print(acc)