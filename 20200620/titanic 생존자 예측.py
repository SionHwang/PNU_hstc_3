import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier

#파일 불러오기
train = pd.read_csv("/titanic/train.csv")
test = pd.read_csv("/titanic/test.csv")
submission= pd.read_csv("/titanic/sample_submission.csv")


#전처리
train["Age"] = train["Age"].fillna(28)
test["Age"] = test["Age"].fillna(28)
train["Embarked"] = train["Embarked"].fillna("S")
train.isna().sum()
train["Sex"]=train["Sex"].map({"male":0,"female":1})


#모델링
X_train = train[['Sex','Pclass','Age']]
y_train = train["Survived"]

test = test[['Sex','Pclass','Age']]
test["Sex"]=test["Sex"].map({"male":0,"female":1})
X_test = test

lr = LogisticRegression()

#dt = DecisionTreeClassifier()


lr.fit(X_train,y_train)
lr.predict(X_test)
lr_pred=lr.predict_proba(X_test)[:,1]
submission["Survived"] = lr_pred

#dt.fit(X_train,y_train)
#dt_pred=dt.predict_proba(X_test)[:,1]
#submission["Survived"] = dt_pred




#출력
submission.to_csv('logistic_regression_pred.csv', index =False)
#submission.to_csv('decision_tree_pred.csv', index =False)
