# -*- coding: utf-8 -*-
"""
Spyder Editor

Kaggle Titantic example from Ultraviolet
"""

import re
import numpy as np
import pandas as pd
import random as rd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA


# read in the training and testing data into Pandas.DataFrame objects

sTrain = "C:\\Users\\Ethan\\Desktop\\Python Project\\kaggle-titanic-tutorial\\data\\raw\\train.csv"
sTest = "C:\\Users\\Ethan\\Desktop\\Python Project\\kaggle-titanic-tutorial\\data\\raw\\test.csv"

input_df = pd.read_csv(sTrain, header=0)
submit_df  = pd.read_csv(sTest,  header=0)

# merge the two DataFrames into one
df = pd.concat([input_df, submit_df])
 
# re-number the combined data set so there aren't duplicate indexes
df.reset_index(inplace=True)
 
# reset_index() generates a new column that we don't want, so let's get rid of it
df.drop('index', axis=1, inplace=True)
 
# the remaining columns need to be reindexed so we can access the first column at '0' instead of '1'
df = df.reindex_axis(input_df.columns, axis=1)
 
print (df.shape[1], "columns:", df.columns.values)
print ("Row count:", df.shape[0])

### Missing Values
# Replace missing values with "U0"
df['Cabin'][df.Cabin.isnull()] = 'U0'

# Take the median of all non-null Fares and use that for all missing values
df.Fare[df.Fare.isnull()] = df.Fare.dropna().median()

# Replace missing values with most common port
df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values

### Populate missing ages  using RandomForestClassifier
def setMissingAges(df):
    
    # Grab all the features that can be included in a Random Forest Regressor
    age_df = df[['Age','Embarked','Fare', 'Parch', 'SibSp', 'Title_id','Pclass','Names','CabinLetter']]
    
    # Split into sets with known and unknown Age values
    knownAge = age_df.loc[ (df.Age.notnull()) ]
    unknownAge = age_df.loc[ (df.Age.isnull()) ]
    
    # All age values are stored in a target array
    y = knownAge.values[:, 0]
    
    # All the other values are stored in the feature array
    X = knownAge.values[:, 1::]
    
    # Create and fit a model
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(X, y)
    
    # Use the fitted model to predict the missing values
    predictedAges = rtr.predict(unknownAge.values[:, 1::])
    
    # Assign those predictions to the full data set
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    
    return df
 ########
    
### Variable Transformations    
# Create a dataframe of dummy variables for each distinct value of 'Embarked'
dummies_df = pd.get_dummies(df['Embarked'])
 
# Rename the columns from 'S', 'C', 'Q' to 'Embarked_S', 'Embarked_C', 'Embarked_Q'
dummies_df = dummies_df.rename(columns=lambda x: 'Embarked_' + str(x))
 
# Add the new variables back to the original data set
df = pd.concat([df, dummies_df], axis=1)
 
# (or written as a one-liner):
df = pd.concat([df, pd.get_dummies(df['Embarked']).rename(columns=lambda x: 'Embarked_' + str(x))], axis=1)

# StandardScaler will subtract the mean from each value then scale to the unit variance
scaler = preprocessing.StandardScaler()
df['Age_scaled'] = scaler.fit_transform(df['Age'])

# Divide all fares into quartiles
df['Fare_bin'] = pd.qcut(df['Fare'], 4)

# qcut() creates a new variable that identifies the quartile range, but we can't use the string so either
# factorize or create dummies from the result
df['Fare_bin_id'] = pd.factorize(df['Fare_bin'])
df = pd.concat([df, pd.get_dummies(df['Fare_bin']).rename(columns=lambda x: 'Fare_' + str(x))], axis=1)

###Derived Variables
# how many different names do they have? 
df['Names'] = df['Name'].map(lambda x: len(re.split(' ', x)))

df['Title'] = df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
 
# Group low-occuring, related titles together
df['Title'][df.Title == 'Jonkheer'] = 'Master'
df['Title'][df.Title.isin(['Ms','Mlle'])] = 'Miss'
df['Title'][df.Title == 'Mme'] = 'Mrs'
df['Title'][df.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
df['Title'][df.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'
 
# Build binary features
df = pd.concat([df, pd.get_dummies(df['Title']).rename(columns=lambda x: 'Title_' + str(x))], axis=1)

# Replace missing values with "U0"
df['Cabin'][df.Cabin.isnull()] = 'U0'
 
# Create a feature for the deck
df['Deck'] = df['Cabin'].map( lambda x : re.compile("([a-zA-Z]+)").search(x).group())
df['Deck'] = pd.factorize(df['Deck'])[0]
 
# Create binary features for each deck
decks = pd.get_dummies(df['Deck']).rename(columns=lambda x: 'Deck_' + str(x))
df = pd.concat([df, decks], axis=1)
 
# Create feature for the room number
df['Room'] = df['Cabin'].map( lambda x : re.compile("([0-9]+)").search(x).group()).astype(int) + 1