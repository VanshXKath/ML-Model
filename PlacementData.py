import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions


#Upload Data
df=pd.read_csv('placement.csv')
#print(df)
#print(df.head())
#print(df.info())

#Clean
df=df.iloc[:,1:]
#print(df.head())

#EDA
plt.scatter(df['cgpa'],df['iq'],c=df['placement'])
#plt.show()

#Extract Input and Output
X = df.iloc[:,0:2]
Y = df.iloc[:,-1]
#print(X)
#print(Y)

#Train Test Split
XS,XT,YS,YT=train_test_split(X,Y,test_size=0.1)
#print(XS)
#print(XT)
#print(YS)
#print(YT)


#Scale the Value(-1 to 1)
scaler = StandardScaler()
XS = scaler.fit_transform(XS)
#print(XS)
XT = scaler.transform(XT)
#print(XT)


#Train Model
clf = LogisticRegression()
clf.fit(XS,YS)
pred = (clf.predict(XT))
#print((YT))

#Accuracy Rate
print(accuracy_score(pred,YT))


#Plot Regression
plot_decision_regions(XS,YS.values,clf=clf, legend=2)
plt.show()
