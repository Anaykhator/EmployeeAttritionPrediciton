# -*- coding: utf-8 -*-
"""
Created on Tue May 19 22:19:37 2020

@author: Dell
"""
from keras.wrappers.scikit_learn import KerasClassifier
import eli5
from eli5.sklearn import PermutationImportance

def data():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score
    import numpy as np
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline
    df = pd.read_csv('original_dataset.csv')
    df = df.drop(
        ['DailyRate', 'EmployeeCount', 'YearsAtCompany', 'TotalWorkingYears', 'JobLevel', 'HourlyRate', 'MonthlyRate',
         'Over18', 'StandardHours', 'EnvironmentSatisfaction', 'JobInvolvement', 'PerformanceRating',
         'TrainingTimesLastYear', 'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance',
         'YearsWithCurrManager'], axis=1)
    df = df[['Attrition', 'Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'Education', 'EducationField',
             'Gender', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked', 'OverTime',
             'PercentSalaryHike', 'YearsInCurrentRole', 'YearsSinceLastPromotion']]
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

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

    X = X.astype(float)
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)

    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [1, 2, 5, 6, 7, 9])],
                                          remainder='passthrough')
    # one  = OneHotEncoder(categories = [1,2,5,6,7,9], drop = 'first', handle_unknown = 'ignore')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.float)

    oversample = SMOTE()
    undersample = RandomUnderSampler()
    steps = [('o', oversample), ('u', undersample)]
    pipeline = Pipeline(steps=steps)
    X, y = pipeline.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=0)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, y_train, X_test, y_test


def build_ntwrk():
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.metrics import Recall
    model = Sequential()
    model.add(Dense(units = 14, activation = 'relu', input_dim = 30))
    model.add(Dropout(0.2995426239200045))
    model.add(Dense(units = 18, activation = 'relu'))
    model.add(Dropout(0.20974398051884063))
    model.add(Dense(units = 1, activation = 'sigmoid'))

    rmsprop = keras.optimizers.RMSprop(lr=10**-2)

    model.compile(optimizer = rmsprop, loss = 'binary_crossentropy', metrics = ['accuracy'])

    result = model.fit(x= X_train,y = y_train, batch_size = 24, epochs = 200, validation_split = 0.2)
    model.save('saved_model.h5')
    print('Model saved')
    model = load_model('saved_model.h5')
    return model


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, recall_score, f1_score
from keras.models import load_model

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
# from keras.metrics import Recall

X_train, y_train, X_test, y_test = data()

saved_model = build_ntwrk()

score, acc = saved_model.evaluate(X_test, y_test)

print(score, acc)

y_pred = saved_model.predict_classes(X_test)

from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print(confusion_matrix(y_test, y_pred))

sens = tp / (tp + fn)
# spec = tn / (tn + fp)
print(sens)
# print(y_pred)
model = KerasClassifier(build_fn=build_ntwrk, batch_size=24, epochs=200)

# perm = PermutationImportance(model, random_state=1).fit(X_train, y_train)
# eli5.show_weights(perm, feature_names=X_train.columns.tolist())

scoring = {'acc': make_scorer(accuracy_score),
           'rec': make_scorer(recall_score),
           'f1': make_scorer(f1_score)}

# cross_val = cross_validate(estimator=model, X=X_train, y=y_train, scoring=scoring, cv=10)
'''
print(cross_val.keys())
print('Accuracy Mean : ', cross_val['test_acc'].mean())
print('Accuracy Std : ', cross_val['test_acc'].std())

print('Recall Mean : ', cross_val['test_rec'].mean())
print('Recall Std : ', cross_val['test_rec'].std())

print('f1_score Mean : ', cross_val['test_f1'].mean())
print('f1_score Std : ', cross_val['test_f1'].std())
'''