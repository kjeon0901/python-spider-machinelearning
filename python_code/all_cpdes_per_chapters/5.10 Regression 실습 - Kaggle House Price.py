import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier



diabetes_data = pd.read_csv('C:/jeon/diabetes.csv')

print(diabetes_data.info()) # null값 x

'''
Outcome 1-당뇨병o, 0-당뇨병x

1. 로지스틱 회귀, LGB, XGB, GBM, 랜덤포레스트 이용
    => 각 estimator 마다 - 정확도, 정밀도, 재현율, f1, roc_auc(predict_proba() 필요함)값 측정
2. 
'''

X = diabetes_data.iloc[:, :-1]
y = diabetes_data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=156, stratify=y)

lr_clf = LogisticRegression()
lgb_clf = LGBMClassifier()
xgb_clf = XGBClassifier()
gbm_clf = GradientBoostingClassifier()
rf_clf = RandomForestClassifier()

models = [lr_clf, lgb_clf, xgb_clf, gbm_clf, rf_clf]
for model in models:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n\n###### ", model, " ######")
    print("정확도 : ", accuracy_score(pred, y_test))
    print("정밀도 : ", precision_score(pred, y_test))
    print("재현율 : ", recall_score(pred, y_test))
    print("f1 : ", f1_score(pred, y_test))
    print("roc_auc : ", roc_auc_score(pred_proba, y_test))
    
    
    
    
    
    
    
    
    
    