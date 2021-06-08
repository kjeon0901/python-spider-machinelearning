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
   그리고 레이블 데이터와 상관계수의 '절댓값'이 가장 높은 3개의 feature데이터가 뭔지 구하기
3. 3개의 feature데이터 오름차순 정렬해서 scatter 찍어보면 outlier 보임. sorting 해야 더 명확히 보임. 
    outlier같은 index는 제거해버리기. 
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

head3=[]
for i in range(3):
    head3.append(X[label_corr.head(3).index[i]].sort_values(ascending=False))

min_x_axle = min(min(head3[0]), min(head3[1]), min(head3[2]))
min_y_axle = max(max(head3[0]), max(head3[1]), max(head3[2]))

plt.scatter(np.linspace(min_x_axle, min_y_axle, num=284807), head3[0].values, marker='o', c=head3[0], s=25, cmap='rainbow', edgecolor='k')
plt.scatter(np.linspace(min_x_axle, min_y_axle, num=284807), head3[1].values, marker='o', c=head3[1], s=25, cmap='rainbow', edgecolor='k')
plt.scatter(np.linspace(min_x_axle, min_y_axle, num=284807), head3[2].values, marker='o', c=head3[2], s=25, cmap='rainbow', edgecolor='k')
'''
quentile (75%, 50%, 25%) 확인해서 최댓값+1.5*q, 최솟값-1.5*q로 이상치 구하는 것보다 이렇게 직접 구하는 게 더 정확할 때 多 !
그래프에서 어떤 값까지 이상치로 볼지는 온전히 내 판단. 
나중에 퍼포먼스 보고 다시 여러번 이상치 걸러서 그렇게 이상치 제거된 최적의 피처데이터 만들면 됨. 
'''

cond1 = head3[0] < -25
cond2 = head3[1] > 10
cond3 = head3[2] > 6

outlier_index = X[ cond1 | cond2 | cond3 ].index # 좌표에서 outlier 좌표만 포함된 범위 선택
X_copy = X.copy()
X_copy = X_copy.drop(outlier_index, axis=0)













