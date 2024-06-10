import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


data=pd.read_csv("wine-quality.csv")

X=data.iloc[:,:-1]
Y=data.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Mean ab error: ", metrics.mean_absolute_error(y_test,y_pred))
print("Mean sq error: ", metrics.mean_squared_error(y_test,y_pred))
print("Root mean sq error: ", (metrics.mean_squared_error(y_test,y_pred))**0.5)