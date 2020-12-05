import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
df=pd.read_csv(r"PATH/project.csv")
#Separating null data
df0=df.dropna(axis=0)
nulldata=df[df.isnull().any(axis=1)]
#Visualization of numeric datatypes for outliers
sns.boxplot(x=df0['ApplicantIncome'])
sns.boxplot(x=df0['CoapplicantIncome'])
sns.boxplot(x=df0['LoanAmount'])
#Finding z-score to remove datapoints outside IQR
df0['ApplicantIncomeScore']=np.abs(stats.zscore(df0['ApplicantIncome']))
df0['CoapplicantIncomeScore']=np.abs(stats.zscore(df0['CoapplicantIncome']))
df0['LoanAmountScore']=np.abs(stats.zscore(df0['LoanAmount'])) 
df1 =df0.loc[(df0['ApplicantIncomeScore'] < 3) & (df0['CoapplicantIncomeScore'] < 3) & (df0['LoanAmountScore'] < 3)]
df1=df1.drop(['ApplicantIncomeScore','CoapplicantIncomeScore','LoanAmountScore','Loan_ID'],axis=1)
#Concatenating nulldata and treated dataframe to fill nan values(All missing data is found to be in categorical datatypes, hence we replace it with mode)
df1=pd.concat([df1,nulldata],axis=0)
df1.fillna(df1.mode().iloc[0],inplace=True)
#Encoding categorical datatypes
encoded=pd.get_dummies(df1.iloc[:,1:6],drop_first=True)
df1['Property_Area']=pd.get_dummies(df['Property_Area'],drop_first=True)
newdata=pd.concat([df1.drop(['Gender','Married','Dependents','Education','Self_Employed'],axis=1),encoded],axis=1)
#Standardising the datapoints
sc=StandardScaler()
X=pd.DataFrame(sc.fit_transform(newdata.drop(['Loan_ID','Loan_Status'],axis=1)),columns=['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','Gender_Male','Married_Yes','Dependents_1','Dependents_2','Dependents_3+','Education_Not Graduate','Self_Employed_Yes'])
# Target class
Y=pd.DataFrame(newdata['Loan_Status'])
#Visualization
sns.pairplot(df1,hue="Loan_Status")
#Splitting data into train and test samples
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
#Running ExhaustiveFeatureSelector() for feature_selection on 3 different classifiers
efs=ExhaustiveFeatureSelector(RandomForestClassifier(),max_features=6,scoring='roc_auc',cv=5)
efs_fit=efs.fit(X_train,Y_train)
selected_features=X_train.columns[list(efs_fit.best_idx_)]
print(selected_features)
print(efs_fit.best_score_)
rClassifier=RandomForestClassifier(random_state=0)
rClassifier.fit(X_train[selected_features],Y_train)
Y_RCF=rClassifier.predict(X_test[selected_features])
print(classification_report(Y_test,Y_RCF))
efs_naive=ExhaustiveFeatureSelector(GaussianNB(),max_features=6,scoring='roc_auc',cv=4)
efs_naive_fit=efs_naive.fit(X_train,Y_train)
selected_features_naive=X_train.columns[list(efs_naive_fit.best_idx_)]
print(selected_features_naive)
print(efs_naive_fit.best_score_)
gNB=GaussianNB()
gNB.fit(X_train[selected_features_naive],Y_train)
Y_gNB=gNB.predict(X_test[selected_features_naive])
print(classification_report(Y_test,Y_gNB))
efs_logistic=ExhaustiveFeatureSelector(LogisticRegression(),max_features=6,scoring='roc_auc',cv=4)
efs_logistic_fit=efs_logistic.fit(X_train,Y_train)
selected_features_logistic=X_train.columns[list(efs_logistic_fit.best_idx_)]
print(selected_features_logistic)
print(efs_logistic_fit.best_score_)
logistic=LogisticRegression()
logistic.fit(X_train[selected_features_logistic],Y_train)
Y_logistic=logistic.predict(X_test[selected_features_logistic])
print(classification_report(Y_test,Y_logistic))
