import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


creditcard_data = pd.read_csv('C:/jeon/creditcard.csv')

print(creditcard_data.info()) # null값 x

'''
Outcome : Class 0-사기X 1-사기O (신용카드)

1. 원본 데이터 프레임 복사 후, 로지스틱회귀와 LGB 이용해서 정확도, 정밀도, 재현율, F1, roc_auc 구하기
2. corr()함수로 데이터 간의 상관관계 계수 구한 후, heatmap 그리기
3. 레이블 데이터와 상관계수의 '절댓값'이 가장 높은 3개의 feature데이터가 뭔지 구하기

'''

X = creditcard_data.iloc[:, :-1]
y = creditcard_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=156, stratify=y)

lr_clf = LogisticRegression(max_iter=200)
lgb_clf = LGBMClassifier()

models = [lr_clf, lgb_clf]
for idx, model in enumerate(models):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]

    print("\n정확도 : {0:.4f}".format(accuracy_score(pred, y_test)))
    print("정밀도 : {0:.4f}".format(precision_score(pred, y_test)))
    print("재현율 : {0:.4f}".format(recall_score(pred, y_test)))
    print("f1 : {0:.4f}".format(f1_score(pred, y_test)))
    print("roc_auc : {0:.4f}".format(roc_auc_score(y_test, pred_proba)))
    
corr = creditcard_data.corr()
sns.heatmap(corr, cmap = 'RdBu')

label_corr = corr.iloc[:-1, -1] # 레이블 데이터와의 상관계수
label_corr = abs(label_corr).sort_values(ascending=False)
print(label_corr.head(3))