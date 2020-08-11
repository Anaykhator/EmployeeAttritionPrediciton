import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score
from flask import flash
# from sklearn import preprocessing
import numpy as np
import pickle
from keras.models import load_model

def check(X, clf):
    print("\n\n\n\n\n\n Checking Model\n\n\n\n\n")
    X = np.array(X)
    labelencoder_X_1 = LabelEncoder()
    X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    labelencoder_X_2 = LabelEncoder()
    X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
    labelencoder_X_5 = LabelEncoder()
    X[:, 5] = labelencoder_X_5.fit_transform(X[:, 5])
    labelencoder_X_6 = LabelEncoder()
    X[:, 6] = labelencoder_X_6.fit_transform(X[:, 6])
    labelencoder_X_7 = LabelEncoder()
    X[:, 7] = labelencoder_X_7.fit_transform(X[:, 7])
    labelencoder_X_9 = LabelEncoder()
    X[:, 9] = labelencoder_X_9.fit_transform(X[:, 9])
    labelencoder_X_12 = LabelEncoder()
    X[:, 12] = labelencoder_X_12.fit_transform(X[:, 12])
    X=X.astype(float)

    print('before one hot encoding', X)

    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [1, 2, 5, 6, 7, 9])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.float)

    print('after one hot encoding', X)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)

    print(clf.summary)
    p = clf.predict_classes(X)
    t = ()
    for x in p:
        if x == 0:
            a = 'No'
        else:
            a = 'Yes'
        t = t+(a,)
    return t


# def analyze(df):
#     intervals = [x for x in range(0, round(df['MonthlyIncome'].max(),-3)+1, 2000)]
#     categories = ['<'+str(x) for x in range(2000, round(df['MonthlyIncome'].max(),-3)+1, 2000)]
#     df1 = df
#     df1['Income_Categories'] = pd.cut(df.MonthlyIncome, intervals, labels=categories)
#     ax = sns.countplot(x="Income_Categories", hue="Attrition", palette="Set1", data=df1)
#     ax.set(title="Monthly Income vs Attrition", xlabel="Income group", ylabel="Total")
#     plt.xticks(rotation=-30)
#     plt.subplots_adjust(left=0.12,bottom=0.18,right=0.90,top=0.88)
#
#     intervals = [x for x in range(18,63,3)]
#     categories = ['<'+str(x) for x in range(21,63,3)]
#     df1 = df
#     df1['Age_Categories'] = pd.cut(df.Age, intervals, labels=categories)
#     ax = sns.countplot(x="Age_Categories", hue="Attrition", palette="Set1", data=df1)
#     ax.set(title="Age vs Attrition", xlabel="Age group", ylabel="Total")
#     plt.xticks(rotation=-30)
#     plt.subplots_adjust(left=0.12,bottom=0.18,right=0.90,top=0.88)
#
#     intervals = [x for x in range(0,round(df['DistanceFromHome'].max(),-1)+1,2)]
#     categories = ['<'+str(x) for x in range(2,round(df['DistanceFromHome'].max(),-1)+1,2)]
#     df1 = df
#     df1['Distance_from_home'] = pd.cut(df.DistanceFromHome, intervals, labels=categories)
#     ax = sns.countplot(x="Distance_from_home", hue="Attrition", palette="Set1", data=df1)
#     ax.set(title="Distance from home vs Attrition", xlabel="Distance", ylabel="Total")
#     plt.xticks(rotation=-30)
#     plt.subplots_adjust(left=0.12,bottom=0.18,right=0.90,top=0.88)
#
#     ax = sns.countplot(x="PercentSalaryHike", hue="Attrition", palette="Set1", data=df1)
#     ax.set(title="Salary Hike Percentage vs Attrition", xlabel="Salary Hike Percentage", ylabel="Total")
#     plt.subplots_adjust(left=0.12,bottom=0.18,right=0.90,top=0.88)
#
#     ax = sns.countplot(x="NumCompaniesWorked", hue="Attrition", palette="Set1", data=df1)
#     ax.set(title="Number Of Previously Worked Companies vs Attrition", xlabel="Number Of Previously Worked Companies", ylabel="Total")
#     plt.subplots_adjust(left=0.12,bottom=0.18,right=0.90,top=0.88)
#
#     intervals = [x for x in range(0,22,2)]
#     categories = ['<'+str(x) for x in range(2,22,2)]
#     df1 = df
#     df1['Current_Role'] = pd.cut(df.YearsInCurrentRole, intervals, labels=categories)
#     ax = sns.countplot(x="Current_Role", hue="Attrition", palette="Set1", data=df1)
#     ax.set(title="Number Of Years in Current Role vs Attrition", xlabel="Number Of Years in Current Role", ylabel="Total")
#     plt.xticks(rotation=-30)
#     plt.subplots_adjust(left=0.12,bottom=0.18,right=0.90,top=0.88)
#
#     ax = sns.countplot(x="OverTime", hue="Attrition", palette="Set1", data=df1)
#     ax.set(title="Over Time vs Attrition", xlabel="Over Time", ylabel="Total")
#     plt.subplots_adjust(left=0.12,bottom=0.18,right=0.90,top=0.88)
#
#     ax = sns.countplot(x="JobRole", hue="Attrition", palette="Set1", data=df1)
#     ax.set(title="Job Role vs Attrition", xlabel="Job Role", ylabel="Total")
#     plt.xticks(rotation=70)
#     plt.subplots_adjust(left=0.1,bottom=0.36,right=0.92,top=0.9)
#
#     intervals = [x for x in range(0,18,2)]
#     categories = ['<'+str(x) for x in range(2,18,2)]
#     df1 = df
#     df1['Promotion'] = pd.cut(df.YearsSinceLastPromotion, intervals, labels=categories)
#     ax = sns.countplot(x="Promotion", hue="Attrition", palette="Set1", data=df1)
#     ax.set(title="Number of Years since Promotion vs Attrition", xlabel="Number of Years since Promotion", ylabel="Total")
#     plt.xticks(rotation=-30)
#     plt.subplots_adjust(left=0.12,bottom=0.18,right=0.90,top=0.88)
#
#     ax = sns.countplot(x="MaritalStatus", hue="Attrition", palette="Set1", data=df1)
#     ax.set(title="Marital Status vs Attrition", xlabel="Marital Status", ylabel="Total")
#     plt.subplots_adjust(left=0.12,bottom=0.18,right=0.90,top=0.88)


def run(data):
    # loaded_rf = pickle.load(open('rf.sav', 'rb'))
    print("\n\n\n\n\n\n Loading Model\n\n\n\n\n")
    loaded_rf = load_model('saved_model.h5')
    print("\n\n\n\n\n\n Loaded Model\n\n\n\n\n")
    X = [list(elem) for elem in data]
    [r.pop(0) for r in X]
    att = check(X, loaded_rf)
    i = 0
    for row in att:
        X[i].insert(0, row)
        i = i+1
    df1 = pd.DataFrame(X)
    df1.columns=['Attrition', 'Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'Gender', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 'YearsInCurrentRole', 'YearsSinceLastPromotion']
    # analyze(df, loaded_rf)
    df1.to_csv('dataset1.csv')
    return att
