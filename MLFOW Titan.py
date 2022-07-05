# Databricks notebook source
import pandas as pd
import numpy as np

# COMMAND ----------

file_location = "/FileStore/tables/titan_train.csv"

# COMMAND ----------

import pandas as pd

df_titan = pd.read_csv("/dbfs/FileStore/shared_uploads/abdul.aleem@mdlz.com/titan_train-1.csv")

# COMMAND ----------

df_titan.head()

# COMMAND ----------

df_titan.dtypes

# COMMAND ----------

#check the missing values
df_titan.isna().sum()

# COMMAND ----------

df_titan[df_titan["Age"]=="?"]["Age"]

# COMMAND ----------

df_titan["Embarked"].value_counts()

# COMMAND ----------

df_titan.shape

# COMMAND ----------

# replacing the missing values in Age column
import seaborn as sns

# COMMAND ----------

#to filter the not null values
age=df_titan[df_titan["Age"].isna()==False]["Age"]
age.dtypes
age

# COMMAND ----------

import matplotlib.pyplot as plt
age.plot.density()

# COMMAND ----------

sns.distplot(age)

# COMMAND ----------

from scipy.stats import skew
skew(age)

# COMMAND ----------

# filling the missing values with mean
m=np.mean(age)
m

# COMMAND ----------

df_titan["Age"].fillna(m,inplace=True)

# COMMAND ----------

df_titan.isna().sum()

# COMMAND ----------

df_titan.dtypes

# COMMAND ----------

df_titan["Cabin"].value_counts()

# COMMAND ----------

#converting the column into number
df_titan["Cabin"]=df_titan["Cabin"].str.extract("(\d+)")

# COMMAND ----------

df_titan["Cabin"]

# COMMAND ----------

df_titan.isna().sum()

# COMMAND ----------

#checking the embarked column d=for missing values
df_titan["Embarked"].value_counts()

# COMMAND ----------

#filling the missing values 
df_titan["Embarked"].fillna("S",inplace=True)

# COMMAND ----------

# dropping the columns Name, PassengerId, Cabin
df_titan.drop(["PassengerId","Name","Cabin" ],axis=1,inplace=True)

# COMMAND ----------

df_titan.isna().sum()

# COMMAND ----------

df_titan.dtypes

# COMMAND ----------

# Analysing the ticket column
df_titan["Ticket"]

# COMMAND ----------

# extracting the numbers from ticket
df_titan["Ticket"]=df_titan["Ticket"].str.extract("(\d+)")

# COMMAND ----------

df_titan["Ticket"]

# COMMAND ----------

df_titan.isna().sum()

# COMMAND ----------

#removing 4 rows
df_titan=df_titan.dropna()

# COMMAND ----------

df_titan.isna().sum()

# COMMAND ----------

df_titan.dtypes

# COMMAND ----------

# converting Ticket into number
df_titan["Ticket"]=pd.to_numeric(df_titan["Ticket"])

# COMMAND ----------

#Performing one hot encoding with sex and Embarked
df_titan=pd.get_dummies(df_titan,columns=["Sex","Embarked"])

# COMMAND ----------

df_titan.dtypes

# COMMAND ----------

df_titan.head()

# COMMAND ----------

df_titan.columns

# COMMAND ----------

#creating x and y
x=df_titan.loc[:,['Pclass', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare',
       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]

# COMMAND ----------

y=df_titan.loc[:,"Survived"]

# COMMAND ----------

#splitting the dataset
from sklearn.model_selection import train_test_split
x_tr,x_test,y_tr,y_test=train_test_split(x,y,test_size=.2)

# COMMAND ----------

#scaling or normalization

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

# COMMAND ----------

scaled_x_tr=StandardScaler().fit_transform(x_tr)

# COMMAND ----------

#to remove and nan or inf values
scaled_x_tr=np.nan_to_num(scaled_x_tr)

# COMMAND ----------

scaled_x_test=StandardScaler().fit_transform(x_test)

# COMMAND ----------

scaled_x_test=np.nan_to_num(scaled_x_test)

# COMMAND ----------

#finding the optimal value of k

# COMMAND ----------

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# COMMAND ----------

params={"n_neighbors":range(1,11)}

# COMMAND ----------

cv_mod=GridSearchCV(KNeighborsClassifier(),params)

# COMMAND ----------

cv_mod.fit(scaled_x_tr,y_tr)

# COMMAND ----------

cv_mod.best_params_

# COMMAND ----------

#create the model with k=10
knn_mod=KNeighborsClassifier(n_neighbors=10).fit(scaled_x_tr,y_tr)

# COMMAND ----------

#prediction from model
p=knn_mod.predict(scaled_x_test)

# COMMAND ----------

from sklearn.metrics import confusion_matrix

# COMMAND ----------

confusion_matrix(y_test,p)

# COMMAND ----------

from sklearn.metrics import accuracy_score

# COMMAND ----------

accuracy_score(y_test,p)

# COMMAND ----------

y_test.shape

# COMMAND ----------

len(p)

# COMMAND ----------

error=[]
for i in range(1,51):
    kmod=KNeighborsClassifier(i).fit(scaled_x_tr,y_tr)
    p=kmod.predict(scaled_x_test)
    error.append(1-accuracy_score(y_test,p))

# COMMAND ----------

sns.relplot(x=range(1,51),y=error,kind="line",marker="o",markerfacecolor="red")

# COMMAND ----------

knn_mod2=KNeighborsClassifier(n_neighbors=21).fit(scaled_x_tr,y_tr)

# COMMAND ----------

p2=knn_mod2.predict(scaled_x_test)

# COMMAND ----------

accuracy_score(y_test,p2)
