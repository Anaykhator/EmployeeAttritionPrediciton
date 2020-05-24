import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

# def analyze(df, clf):
#     feature_importances = pd.DataFrame(clf.feature_importances_, index=['Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'Gender', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 'YearsInCurrentRole', 'YearsSinceLastPromotion'],columns=['importance']).sort_values('importance',ascending=False)
#     feature_importances['x1'] = feature_importances.index
#     ax = feature_importances.plot.bar(x='x1', y='importance', rot=90)
#     plt.savefig('templates/graphs/raw/feature_importances.png', frameon=True)
#
#     intervals = [x for x in range(0, 22000, 2000)]
#     categories = ['<'+str(x) for x in range(2000, 22000, 2000)]
#     df1 = df
#     df1['Income_Categories'] = pd.cut(df.MonthlyIncome, intervals, labels=categories)
#     ax = sns.countplot(x="Income_Categories", hue="Attrition", palette="Set1", data=df1)
#     ax.set(title="Monthly Income vs Attrition", xlabel="Income group", ylabel="Total")
#     plt.xticks(rotation=-30)
#     plt.show()
#     # plt.savefig('templates/graphs/raw/MIvsAttr.png')
#
#     # sns.scatterplot(x="MonthlyIncome", hue="Attrition", palette="Set1", data=df)
#     # ax.set(title="Monthly Income vs Attrition", xlabel="Income group", ylabel="Total")
#     # sns.pairplot(data=df,hue='Attrition')
#
#
#     intervals = [x for x in range(18,63,3)]
#     categories = ['<'+str(x) for x in range(21,63,3)]
#     df1 = df
#     df1['Age_Categories'] = pd.cut(df.Age, intervals, labels=categories)
#     ax = sns.countplot(x="Age_Categories", hue="Attrition", palette="Set1", data=df1)
#     ax.set(title="Age vs Attrition", xlabel="Age group", ylabel="Total")
#     plt.xticks(rotation=-30)
#     plt.show()
#     # plt.savefig('templates/graphs/raw/AgevsAttr.png')
#
#     intervals = [x for x in range(0,32,2)]
#     categories = ['<'+str(x) for x in range(2,32,2)]
#     df1 = df
#     df1['Distance_from_home'] = pd.cut(df.DistanceFromHome, intervals, labels=categories)
#     ax = sns.countplot(x="Distance_from_home", hue="Attrition", palette="Set1", data=df1)
#     ax.set(title="Distance from home vs Attrition", xlabel="Distance", ylabel="Total")
#     plt.xticks(rotation=-30)
#     plt.show()
#     # plt.savefig('templates/graphs/raw/DistanceFromHomevsAttr.png')
#
#     ax = sns.countplot(x="PercentSalaryHike", hue="Attrition", palette="Set1", data=df1)
#     ax.set(title="Salary Hike Percentage vs Attrition", xlabel="Salary Hike Percentage", ylabel="Total")
#     plt.show()
#     # plt.savefig('templates/graphs/raw/PercentSalaryHikevsAttr.png')
#
#     ax = sns.countplot(x="NumCompaniesWorked", hue="Attrition", palette="Set1", data=df1)
#     ax.set(title="Number Of Previously Worked Companies vs Attrition", xlabel="Number Of Previously Worked Companies", ylabel="Total")
#     plt.show()
#     # plt.savefig('templates/graphs/raw/NPWCvsAttr.png')
#
#     intervals = [x for x in range(0,22,2)]
#     categories = ['<'+str(x) for x in range(2,22,2)]
#     df1 = df
#     df1['Current_Role'] = pd.cut(df.YearsInCurrentRole, intervals, labels=categories)
#     ax = sns.countplot(x="Current_Role", hue="Attrition", palette="Set1", data=df1)
#     ax.set(title="Number Of Years in Current Role vs Attrition", xlabel="Number Of Years in Current Role", ylabel="Total")
#     plt.xticks(rotation=-30)
#     plt.show()
#     # plt.savefig('templates/graphs/raw/YICRvsAttr.png')
#
#     ax = sns.countplot(x="OverTime", hue="Attrition", palette="Set1", data=df1)
#     ax.set(title="Over Time vs Attrition", xlabel="Over Time", ylabel="Total")
#     plt.show()
#     # plt.savefig('templates/graphs/raw/OverTimevsAttr.png')
#
#     ax = sns.countplot(x="JobRole", hue="Attrition", palette="Set1", data=df1)
#     ax.set(title="Job Role vs Attrition", xlabel="Job Role", ylabel="Total")
#     plt.xticks(rotation=70)
#     # plt.savefig('static/JobRolevsAttr.png', bbox_inches='tight')
#     # plt.tight_layout(h_pad=0.5)
#     plt.subplots_adjust(left=0.1,bottom=0.36,right=0.92,top=0.9)
#     plt.show()
#
#
#     intervals = [x for x in range(0,18,2)]
#     categories = ['<'+str(x) for x in range(2,18,2)]
#     df1 = df
#     df1['Promotion'] = pd.cut(df.YearsSinceLastPromotion, intervals, labels=categories)
#     ax = sns.countplot(x="Promotion", hue="Attrition", palette="Set1", data=df1)
#     ax.set(title="Number of Years since Promotion vs Attrition", xlabel="Number of Years since Promotion", ylabel="Total")
#     plt.xticks(rotation=-30)
#     plt.show()
#     # plt.savefig('templates/graphs/raw/YSCPvsAttr.png')
#
#     ax = sns.countplot(x="MaritalStatus", hue="Attrition", palette="Set1", data=df1)
#     ax.set(title="Marital Status vs Attrition", xlabel="Marital Status", ylabel="Total")
#     plt.show()
#     # plt.savefig('templates/graphs/raw/MSvsAttr.png')
#
# df3=pd.read_csv('dataset1.csv')
df = pd.read_csv('original_dataset.csv')
df = df.drop(['DailyRate', 'EmployeeCount', 'YearsAtCompany', 'TotalWorkingYears', 'JobLevel', 'HourlyRate', 'MonthlyRate', 'Over18', 'StandardHours', 'EnvironmentSatisfaction', 'JobInvolvement', 'PerformanceRating', 'TrainingTimesLastYear', 'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance', 'YearsWithCurrManager'], axis=1)
df = df[['Attrition', 'Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'Gender', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 'YearsInCurrentRole', 'YearsSinceLastPromotion']]
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

# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40,random_state=0)

# rf = RandomForestClassifier(n_estimators=200)
# rf.fit(X_train,y_train)
# intervals = [x for x in range(0,round(df['DistanceFromHome'].max(),-1)+1,2)]
# categories = ['<'+str(x) for x in range(2,round(df['DistanceFromHome'].max(),-1)+1,2)]
# df1 = df
# df1['Distance_from_home'] = pd.cut(df.DistanceFromHome, intervals, labels=categories)
# ax = sns.countplot(x="Distance_from_home", hue="Attrition", palette="Set1", data=df1)
# ax.set(title="Distance from home vs Attrition", xlabel="Distance", ylabel="Total")
# plt.xticks(rotation=-30)
# plt.show()
# analyze(df3,rf)
# pickle.dump(rf, open('rf.sav', 'wb'))

# xgb = XGBClassifier()
# xgb.fit(X_train, y_train)
# pickle.dump(xgb, open('xgb.sav', 'wb'))
lr = LogisticRegression(class_weight='balanced',solver='lbfgs')
lr.fit(X_train, y_train)
pickle.dump(lr, open('rf.sav', 'wb'))
# vot = VotingClassifier(estimators=[('lr',lr),('rf', rf), ('xg', xgb)], voting='hard')
# vot.fit(X_train,y_train)
final = lr.predict(X_test)
# pickle.dump(vot, open('xgb.sav', 'wb'))
# acc = accuracy_score(y_test, final)
# pre = precision_score(y_test, final)
rec = recall_score(y_test, final)
print(rec)
# y_pred_lr = lr.predict(x_test_lr)
