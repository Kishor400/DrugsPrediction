import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=pd.read_csv("drug200.csv")


Xd=["Age","Sex","BP","Cholesterol","Na_to_K"]


def DataCleansing(data):
	
	data["Sex"]=data["Sex"].replace({"F":0,"M":1})
	data["BP"]=data["BP"].replace({"LOW":0,"NORMAL":1,"HIGH":2})
	data["Cholesterol"]=data["Cholesterol"].replace({"LOW":0,"NORMAL":1,"HIGH":2})
	
	return data
	
	
data=DataCleansing(data)

In=["Age","Sex","BP","Cholesterol","Na_to_K"]

X=data[In]
y=data["Drug"]

X_Train,X_Test,y_Train,y_Test=train_test_split(X,y,test_size=0.2,random_state=42)

model=KNeighborsClassifier()

model.fit(X_Train,y_Train)

yp=model.predict(X_Test)

acc=accuracy_score(yp,y_Test)

print("Accuracy : ",acc*100,"%")

xt=[]
for i in In:
	x=int(input("Enter "+i+" : "))
	xt.append(x)
	
yt=model.predict([xt])
print(yt)