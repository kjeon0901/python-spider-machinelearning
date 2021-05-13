import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler, Binarizer
from sklearn.linear_model import LogisticRegression

diabetes_df = pd.read_csv('C:/jeon/diabetes.csv')
print(diabetes_df['Outcome'].value_counts())
diabetes_df.info( )



each_mean=[]
cng_threshold=False
#정확도, 정밀도, 재현율, F1, AUC 구하기 
def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix( y_test, pred)
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    f1 = f1_score(y_test,pred)
    # ROC-AUC 추가 
    roc_auc = roc_auc_score(y_test, pred_proba)
    print('오차 행렬')
    print(confusion)
    # ROC-AUC print 추가
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))
    if cng_threshold:
        mean = (accuracy + precision + recall + f1 + roc_auc)/5
        each_mean.append(mean)

X = diabetes_df.iloc[:, :-1]
y = diabetes_df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 156)

# 로지스틱 회귀로 학습,예측 및 평가 수행. 
lr_clf = LogisticRegression()
lr_clf.fit(X_train , y_train)
pred = lr_clf.predict(X_test)
# roc_auc_score 수정에 따른 추가
pred_proba = lr_clf.predict_proba(X_test)[:, 1]
get_clf_eval(y_test , pred, pred_proba)



#precisions, recalls 그래프그리기
precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba) # 이 세 개의 size가 다름. 가장 작은 thresholds값까지만 그려야 함
plt.plot(thresholds, precisions[0:thresholds.shape[0]]) #plt.plot(x축, y축)
plt.plot(thresholds, recalls[0:thresholds.shape[0]]) #x축 값이 143, y축 값이 144로 다름. x크기만큼 y크기 정해줘야 오류안남
plt.xlabel('Threshold value'); plt.ylabel('Precision, Recall')



#['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']각 컬럼에서 0값이 차지하는 비율 소수점 둘째자리까지
check_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for x in check_columns:
    new_column = X[x].astype(bool)
    cnt=0
    for idx, i in enumerate(new_column):
        if i==False:
            cnt+=1
            X[x][idx] = X[x].agg('mean')
    print(x, '에서 0이 차지하는 비율: ', round((cnt / new_column.shape[0])*100 , 2))



#StandardScaler - 평균 0 분산 1 표준화 지원
X = diabetes_df.iloc[:, :-1]
y = diabetes_df.iloc[:, -1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 156, stratify=y)



#logisticRegression 이용 학습 및 예측
lr_clf = LogisticRegression()
lr_clf.fit(X_train , y_train)
pred = lr_clf.predict(X_test) # 1 0 1 0 .. 예측값
pred_proba = lr_clf.predict_proba(X_test)[:, 1] #확률값


def get_eval_by_threshold(y_test , pred_proba_c1, thresholds):
    # thresholds list객체내의 값을 차례로 iteration하면서 Evaluation 수행.
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1) 
        custom_predict = binarizer.transform(pred_proba_c1)
        get_clf_eval(y_test, custom_predict, pred_proba_c1)
    

cng_threshold=True
thresholds = [0.3, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.50]
get_eval_by_threshold(y_test ,pred_proba.reshape(-1,1), thresholds) #reshape안해주면 오류뜸. 걍 해주기
print(round(max(each_mean), 4))
cng_threshold=False

'''

[선생님 풀이]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

diabetes_df = pd.read_csv('E:/no1/cho/archive/diabetes.csv')
print(diabetes_df['Outcome'].value_counts())

X = diabetes_df.drop('Outcome', axis = 1 )
y = diabetes_df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,\
                                                    random_state = 156, stratify=y)

lr_clf = LogisticRegression()
lr_clf.fit(X_train , y_train)
pred = lr_clf.predict(X_test)

#accuracy = accuracy_score(y_test , pred)
#precision = precision_score(y_test , pred)
#recall = recall_score(y_test , pred)

pred_proba_c1 = lr_clf.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve( y_test, pred_proba_c1)

print(len(thresholds))

plt.plot(thresholds , precisions[ 0:len(thresholds)] )
plt.plot(thresholds , recalls[ 0:len(thresholds)] )

zero_features = ['Glucose', 'BloodPressure','SkinThickness','Insulin','BMI']

for row in zero_features:
    temp = X[X[row]==0][row]
    print(round((temp.shape[0]/X.shape[0]) * 100 , 2))
    X[row] = X[row].replace(0, X[row].mean())
    
for row in zero_features:
    temp = X[X[row]==0][row]
    print(round((temp.shape[0]/X.shape[0]) * 100 , 2))

'''