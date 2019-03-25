# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Load the training and test sets
train_df = pd.read_csv("input/train.csv")
test_df = pd.read_csv("input/test.csv")

# Format data appropriately, impute missing data, and do feature engineering
def cleanData(df):
    # Create a Family_size feature
    df.loc[:,'Family_size'] = df.Parch + df.SibSp + 1
    
    # Recode Pclass as a categorical
    df.Pclass = df.Pclass.astype('str')
     
    # Extract title from name
    title_ex = "(.*, )|(\..*)"
    df.loc[:,'Title'] = df.Name.apply(lambda x : re.sub(title_ex,"",x))
    
    # group rarely occuring titles
    noble = ['Capt', 'Col', 'Don', 'Jonkheer', 'Lady', 'Major', 'Mlle', 'Mme', 'Sir', 'the Countess', 'Dona']
    df.loc[:,'Title'] = df.Title.apply(lambda x : 'Noble' if x in noble else x)
    df.Title.replace('Ms', 'Miss', inplace=True)
    
    # Extract the Deck letter
    df.loc[:, 'Deck'] = df.Cabin.str[0]
    
    # Get rid of 'T' deck which is rarely occuring
    df.Deck.replace('T', np.nan, inplace=True)

    # Fill in the single missing fare with the mean
    df.Fare.replace(np.nan, df.Fare.mean(), inplace = True)
    
    # Adjust fare.
    df.loc[:, 'Adjusted_fare'] = df.groupby('Ticket', group_keys=False).Fare.transform(lambda x : x.mean() / x.count())
    
    ### Impute missing data
    # Impute missing ages
    age_lookup = df.groupby('Title').Age.mean().to_dict()

    def transform(row):
        if np.isnan(row.Age):
            row.Age = age_lookup[row.Title]
        return row

    df = df.apply(transform, axis = 1)
    
    # Impute deck using SVM
    y = df.Deck[~df.Deck.isna()]
    X = pd.get_dummies(df.loc[~df.Deck.isna(),['Adjusted_fare', 'Pclass', 'Title', 'Embarked', 'Sex']],drop_first=True)

    # Note that title = 'Rev' does not exist in X 
    # Add this in manually to allow for prediction below
    X['Title_Rev'] = 0

    # Model was evaluated with code below - CV accuracy of 0.50
    # Best C: 8.79
    # Best gamma: 0.56
    #    Cs = np.linspace(0.1,10,50)
    #    gammas = np.linspace(0.01, 1,50)
    #    param_grid = {'C': Cs, 'gamma' : gammas}
    #    
    #    grid_search = GridSearchCV(svc, param_grid, cv=5,iid=False)
    #    grid_search.fit(X, y)
    #    
    #    print(grid_search.best_params_)
    #    print(grid_search.best_score_)
    #        
    #    deck_pred = grid_search.predict(X)
    #    
    #    print(confusion_matrix(y, deck_pred))
    #    print(classification_report(y, deck_pred))
    #    print(accuracy_score(y, deck_pred))
    svc = SVC(kernel='rbf', C = 8.79, gamma = 0.56)

    svc.fit(X,y)

    X_missing = pd.get_dummies(df.loc[df.Deck.isna(),['Adjusted_fare', 'Pclass', 'Title', 'Embarked','Sex']],drop_first=True)
    df.loc[df.Deck.isna(),'Deck'] = svc.predict(X_missing)
    
    # Impute missing Embarked to the most frequent class, 'S'
    df.Embarked.replace(np.nan,'S', inplace = True)
    
    # Add a Child variable - defined as under 13
    # The threshold of 13 for child was based on the usage of the honorific master
    df.loc[:,'Child'] = df.Age.apply(lambda x : 1 if x < 13 else 0) 
    
    # Add a KnownDead variable that counts the number of people on a ticket who we know did not survive
    # This does not include one's self to prevent biasing of the Survived variable
    df['KnownDead'] = df.groupby('Ticket', group_keys=False).Survived.transform(lambda x : x.count() + x.isna().sum()  - sum(x != 0))
    df.loc[:, 'KnownDead'] = df.apply(lambda x : x.KnownDead - 1 if x.Survived == 0 else x.KnownDead ,axis=1)
    
    # Remove unused columns
    df.drop(['Cabin','Name','Ticket'], inplace = True, axis = 1)
    
    # return the separate training and test sets
    return [df[df.Type == 'train'], df[df.Type == 'test']]

train_df.loc[:, 'Type'] = 'train'
test_df.loc[:, 'Type'] = 'test'

impute_set = pd.concat([train_df, test_df], sort = False)

train_clean, test_clean = cleanData(impute_set)

# Prep data for fitting
y = train_clean.Survived
X = pd.get_dummies(train_clean.loc[:,['Family_size','Adjusted_fare', 'Pclass', 'Title', 'Embarked', 'Sex','Age','Child','KnownDead']],drop_first=True)

X.Family_size = scale(X.Family_size)
X.Age = scale(X.Age)
X.Adjusted_fare = scale(X.Adjusted_fare)
X.KnownDead = scale(X.KnownDead)

# Prep submission set for prediction
X_submission = pd.get_dummies(test_clean.loc[:,['Family_size','Adjusted_fare', 'Pclass', 'Title', 'Embarked', 'Sex','Age','Child','KnownDead']],drop_first=True)

X_submission.Family_size = scale(X_submission.Family_size)
X_submission.Age = scale(X_submission.Age)
X_submission.Adjusted_fare = scale(X_submission.Adjusted_fare)
X_submission.KnownDead = scale(X_submission.KnownDead)

# Fit a logistic regression model to the data
logreg = LogisticRegression()
logreg_cv_scores = cross_val_score(logreg, X, y, cv = 5)

print(logreg_cv_scores.mean(), logreg_cv_scores.std())

logreg.fit(X, y)

# Create logisitic regression submission - 0.76555
logreg_submission = pd.concat([test_df.PassengerId, pd.DataFrame(logreg.predict(X_submission)).astype(int)], axis = 1)
logreg_submission.columns = ['PassengerId','Survived']

logreg_submission.to_csv('submissions/logreg_submission.csv', index = False)

# Fit a RidgeClassifier model - 0.76555
alpha = np.logspace(-1,5,100)
ridge_param_grid = {'alpha': alpha}

ridge = RidgeClassifier()
ridge_grid_search = GridSearchCV(ridge, ridge_param_grid, cv=5,iid=False)
ridge_grid_search.fit(X, y)    
        
print(ridge_grid_search.best_params_)
print(ridge_grid_search.best_score_)

ridge_submission = pd.concat([test_df.PassengerId, pd.DataFrame(ridge_grid_search.predict(X_submission).astype(int))], axis = 1)
ridge_submission.columns = ['PassengerId','Survived']

# How different is this from logisitic regression
np.sum(abs(logreg_submission.Survived - ridge_submission.Survived))

# Create the submission file
ridge_submission.to_csv('submissions/ridge_submission.csv', index = False)

# Create an SVM submission - 0.77033
Cs = np.logspace(-2,5,10)
gammas = np.logspace(-2,5,10)
svc_param_grid = {'C': Cs, 'gamma' : gammas}
 
svc = SVC()
svc_grid_search = GridSearchCV(svc, svc_param_grid, cv=5,iid=False)
svc_grid_search.fit(X, y)    
        
print(svc_grid_search.best_params_)
print(svc_grid_search.best_score_)

svc_submission = pd.concat([test_df.PassengerId, pd.DataFrame(svc_grid_search.predict(X_submission).astype(int))], axis = 1)
svc_submission.columns = ['PassengerId','Survived']

# How different is this from logisitic regression
np.sum(abs(logreg_submission.Survived - svc_submission.Survived))

# Create the submission file
svc_submission.to_csv('submissions/svc_submission.csv', index = False)

# Create a Random Forest Submission - 0.78468
rf = RandomForestClassifier(n_estimators = 5000, random_state = 1)

rf_cv_scores = cross_val_score(rf, X, y, cv = 5)

print(rf_cv_scores.mean(), rf_cv_scores.std())

rf.fit(X,y)
        
rf_submission = pd.concat([test_df.PassengerId, pd.DataFrame(rf.predict(X_submission).astype(int))], axis = 1)
rf_submission.columns = ['PassengerId','Survived']

# How different is this from logisitic regression
np.sum(abs(logreg_submission.Survived - rf_submission.Survived))

# Create the submission file
rf_submission.to_csv('submissions/rf_submission.csv', index = False)

# Create a data frame comparing the various predictions
test_compare = pd.concat([test_clean,pd.DataFrame(ridge_submission.Survived),pd.DataFrame(svc_submission.Survived),pd.DataFrame(rf_submission.Survived)],axis = 1)

# Create an Decision Tree Submission
dc_param_grid = {
    "learning_rate": [0.1],
    "min_samples_split": [8],
    "min_samples_leaf":  [2],
    "max_depth":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "max_features":["log2","sqrt"],
    "subsample": [0.8],
    "n_estimators":[10,40, 70]
    }


dc = GradientBoostingClassifier()
dc_grid_search = GridSearchCV(dc, dc_param_grid, cv=5,iid=False)
dc_grid_search.fit(X, y)    
        
print(dc_grid_search.best_params_)
print(dc_grid_search.best_score_)

dc_submission = pd.concat([test_df.PassengerId, pd.DataFrame(dc_grid_search.predict(X_submission).astype(int))], axis = 1)
dc_submission.columns = ['PassengerId','Survived']

# How different is this from logisitic regression
np.sum(abs(logreg_submission.Survived - dc_submission.Survived))

# Create the submission file
dc_submission.to_csv('submissions/dc_submission.csv', index = False)
