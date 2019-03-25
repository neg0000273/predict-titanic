# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:20:05 2019

@author: Ruben
"""
#imports
import pandas as pd
import numpy as np
from keras.models import Sequential, clone_model
from keras.layers import Dense, Dropout
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
import re

#Load data
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

def keras_cv(model, X, y, cv = 5, epochs=15):
    kf = KFold(n_splits = cv)
    cv_vals = np.zeros(cv)
    
    for i, [train_index, test_index] in enumerate(kf.split(X)):
        cl_model = clone_model(model)
        cl_model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy',metrics=['accuracy'])

        cl_model.fit(X.iloc[train_index,:], y[train_index],epochs=epochs)
        cv_vals[i] = cl_model.evaluate(X.iloc[test_index,:], y[test_index])[1]
        
    return(np.mean(cv_vals))

#Feature engineer, clean, impute, scale
def clean_data(df):
    #Feature engineer 
    # Extract title from name
    title_ex = "(.*, )|(\..*)"
    df.loc[:,'Title'] = df.Name.apply(lambda x : re.sub(title_ex,"",x))
    
    df['KnownDead'] = df.groupby('Ticket', group_keys=False).Survived.transform(lambda x : x.count() + x.isna().sum()  - sum(x != 0))
    df.loc[:, 'KnownDead'] = df.apply(lambda x : x.KnownDead - 1 if x.Survived == 0 else x.KnownDead,axis=1)
    
    # Extract the Deck letter
    df.loc[:, 'Deck'] = df.Cabin.str[0]
    
    #Impute
    # Impute missing ages
    df.loc[df.Age.isnull(),'Age'] = df.Age.mean()
    df.loc[df.Fare.isnull(),'Fare'] = df.Fare.mean()
    df.loc[df.Embarked.isnull(),'Embarked'] = df.Embarked.mode()
    df.loc[df.Embarked.isnull(),'Deck'] = 'Missing'
    
    #drop columns we don't need
    df.drop(['PassengerId','Ticket','Name','Cabin'], axis=1, inplace = True)
    
    #convert categoricals via one-hot encoding
    df = pd.get_dummies(data = df, columns = ['Pclass','Sex','Title','Embarked','Deck'])
    
    #scale
    df.loc[:,'Age'] = scale(df.loc[:,'Age'])
    df.loc[:,'SibSp'] = scale(df.loc[:,'SibSp'])
    df.loc[:,'Parch'] = scale(df.loc[:,'Parch'])
    df.loc[:,'Fare'] = scale(df.loc[:,'Fare'])
    df.loc[:,'KnownDead'] = scale(df.loc[:,'KnownDead'])
    
    return [df[df.Type == 'train'], df[df.Type == 'test']]

train.loc[:,'Type'] = 'train'
test.loc[:,'Type'] = 'test'

combined = pd.concat([train, test], sort=False)

#Data exploration ??? do this on your own
#check for missing data
combined.isnull().sum()

train_clean, test_clean = clean_data(combined)

train_clean_target = train_clean.Survived
train_clean.drop(['Type','Survived'],axis=1, inplace = True)
test_clean.drop(['Type','Survived'], axis=1, inplace = True)

#Build model
model = Sequential()
model.add(Dense(512, activation = 'relu', input_shape = (train_clean.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))

print(keras_cv(model, X=train_clean, y=train_clean_target, cv=5,epochs=20))

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy',metrics=['accuracy'])
model.fit(train_clean, train_clean_target, epochs=20)

#create submission
submission = pd.concat([test.PassengerId, pd.DataFrame(np.where(model.predict(test_clean) > 0.5,1,0))], axis=1)
submission.columns = ['PassengerId','Survived']
submission.to_csv('submissions/submission_deep_learning.csv', index = False)
