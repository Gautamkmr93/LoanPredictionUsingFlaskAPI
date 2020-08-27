# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 14:34:07 2020

@author: gautam.y.kumar
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from sklearn.linear_model import LogisticRegression
import pickle

traindata=pd.read_csv('C:\\Users\\gautam.y.kumar\\Desktop\\ml project\\Loan_Prediction\\Train.csv')
#handling the NA Values
#print(traindata.isna().sum())
traindata['Gender'] = traindata['Gender'].fillna(traindata['Gender'].value_counts().index[0])
traindata['Married'] = traindata['Married'].fillna(traindata['Married'].value_counts().index[0])
traindata['Dependents']=traindata['Dependents'].fillna(traindata['Dependents'].value_counts().index[0])
traindata['Self_Employed']=traindata['Self_Employed'].fillna(traindata['Self_Employed'].value_counts().index[0])
traindata['LoanAmount'] = traindata['LoanAmount'].fillna(traindata['LoanAmount'].mean())
traindata['Loan_Amount_Term']=traindata['Loan_Amount_Term'].fillna(traindata['Loan_Amount_Term'].mean())
traindata['Credit_History']=traindata['Credit_History'].fillna(traindata['Credit_History'].mean())
#print(traindata.isna().sum())

#handling the categorical varible data
traindata['Gender']=pd.get_dummies(traindata['Gender'])
traindata['Married']=pd.get_dummies(traindata['Married'])
traindata['Dependents']=pd.get_dummies(traindata['Dependents'])
traindata['Education']=pd.get_dummies(traindata['Education'])
traindata['Self_Employed']=pd.get_dummies(traindata['Self_Employed'])
traindata['Property_Area']=pd.get_dummies(traindata['Property_Area'])
traindata['Loan_Status']=pd.get_dummies(traindata['Loan_Status'])

#testdata=pd.read_csv('C:\\Users\\gautam.y.kumar\\Desktop\\ml project\\Loan_Prediction\\Test.csv')
#print(traindata)

#visualization
grid = sns.FacetGrid(traindata, col='Loan_Status')
grid.map(plt.hist, 'Gender')

traindata=traindata.drop('Loan_ID',axis=1)
x=traindata.drop('Loan_Status',axis=1)
y=traindata['Loan_Status']

data_corr = pd.concat([x, y], axis=1)
corr = data_corr.corr()
plt.figure(figsize=(11,7))
sns.heatmap(corr, annot=True)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)

createModel=LogisticRegression()
createModel.fit(x_train,y_train)

pickle.dump(createModel, open('mymodel.pkl','wb'))
model = pickle.load(open('mymodel.pkl','rb'))

print('training has been completed and  its dump into pkl files')

#prediction=createModel.predict(x_test)
#print(prediction)
#score=createModel.score(x_test,y_test)
#print(score)

