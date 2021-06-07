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

2. Glucose, BloodPressure, Insulin의 0 -> 평균값으로 바꿈
SkinTickness의 0 ->  임신횟수(Pregnancies) > 0이면 SkinTickness = 25
                                        == 0                 = 15
BMI의 0 ->   혈압(BloodPressure) >= 110 이면 BMI의 quentile 75%값 (describe로 확인)
                            90<= x <110                     50%
                                 < 90                       25%

'''


X = diabetes_data.iloc[:, :-1]
y = diabetes_data.iloc[:, -1]
X_describe = X.describe()

zero_to_mean=['Glucose', 'BloodPressure', 'Insulin']
for each in zero_to_mean:
    mean = int(X[X[each]!=0][each].agg('mean'))
    X[each] = X[each].apply(lambda x:mean if x==0 else x)
X['SkinThickness'] = X['Pregnancies'].apply(lambda x:25 if x>0 else 15)
X['BMI'] = X['BloodPressure'].apply(lambda x:X_describe['BMI']['75%'] if x >= 110 \
                                    else(X_describe['BMI']['50%'] if (x>=90 and x<110) else X_describe['BMI']['25%']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=156, stratify=y)

#아래에서 인수로 추가해준 건 warning 제거해주기 위해 추가함. 안써줘도 상관없음. 
lr_clf = LogisticRegression(max_iter=200)
lgb_clf = LGBMClassifier()
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
gbm_clf = GradientBoostingClassifier()
rf_clf = RandomForestClassifier()

models = [lr_clf, lgb_clf, xgb_clf, gbm_clf, rf_clf]
for idx, model in enumerate(models):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n",end='')
    print(idx+1)
    print("정확도 : {0:.4f}".format(accuracy_score(pred, y_test)))
    print("정밀도 : {0:.4f}".format(precision_score(pred, y_test)))
    print("재현율 : {0:.4f}".format(recall_score(pred, y_test)))
    print("f1 : {0:.4f}".format(f1_score(pred, y_test)))
    print("roc_auc : {0:.4f}".format(roc_auc_score(y_test, pred_proba)))
    

    
    
    
    
    
    
    
    