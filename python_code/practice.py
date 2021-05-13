import numpy as np

np.argmax(np.bincount([0, 0, 1], weights=[0.2, 0.2, 0.6])) # argmax: 가장 큰 포지션에 있는 아이의 idx 리턴, bincount: 유니크한 원소 개수, weights: 가중치
'''1'''
print(np.bincount([0, 0, 1, 3, 3, 3])) # 0→2, 1→1, 2→0, 3→3  -> [2 1 0 3]
print(np.bincount([0, 0, 1])) # 0→2, 1→1  -> [2 1]
print(np.bincount([0, 0, 1], weights=[0.2, 0.2, 0.6])) #각 포지션의 weight을 부여하면 0→0.4, 1→0.6  -> [0.4 0.6]
print(np.bincount([0, 0, 1, 3, 3], weights=[0.2, 0.3, 0.6, 0.1, 0.1])) # -> [0.5 0.6 0.  0.2]


ex = np.array([[0.9, 0.1],
               [0.8, 0.2],
               [0.4, 0.6]])

p = np.average(ex, axis=0, weights=[0.2, 0.2, 0.6]) #axis=0 : 보통 row방향(아래방향) => 그냥 평균 내면 [0.7 0.3]
print(p)    #가중치를 냈으니까 가중평균 -> 0.9*0.2, 0.8*0.2, 0.4*0.6 더해주면 끝(가중치 자체가 확률이니까) => [0.58 0.42]
p = np.average(ex, axis=0)
print(p)


from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators #__init__에서 사용. estimator 넣어주면 상세 정보 list(zip(names, estimators))를 리턴
import numpy as np
import operator


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin): #BaseEstimator 상속 -> MajorityVoteClassifier 클래스를 커스터마이징한 estimator처럼 사용 가능 (fit, predict, predict_proba 메소드 보면 알수있듯)
    def __init__(self, classifiers, vote='classlabel', weights=None):

        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)} #딕셔너리
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError("vote는 'probability' 또는 'classlabel'이어야 합니다."
                             "; (vote=%r)이 입력되었습니다."
                             % self.vote)

        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError('분류기 개수와 가중치 개수는 동일해야 합니다.'
                             '; %d개의 가중치와, %d개의 분류기가 입력되었습니다.'
                             % (len(self.weights), len(self.classifiers)))

        # self.predict 메서드에서 np.argmax를 호출할 때 
        # 클래스 레이블이 0부터 시작되어야 하므로 LabelEncoder를 사용합니다. ->레이블인코딩 : LabelEncoder를 객체로 생성한 후 , fit( ) 과 transform( ) 으로 label 인코딩 수행. 
        self.lablenc_ = LabelEncoder() #밖에서 우리가 labelencoding 해줬기 때문에 사실 필요 없는데 그냥 해줌
        self.lablenc_.fit(y) #y: 아래에서 붖꽃데이터 test데이터 레이블인코딩 끝내고 더 아래에서 train_test_split해줘서 나온 y_train 데이터. 50개이고 비율은 0:1=1:1 
        self.classes_ = self.lablenc_.classes_ #레이블 인코딩 된 아이의 유니크한 값이 들어감. 여기선 0, 1 두개가 들어감
        self.classifiers_ = []
        for clf in self.classifiers: #클래스 객체 만들 때 받아온 estimator 3개 담긴 리스트. dtype:list, 각각의 요소 dtype:estimator
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y)) #clone():원본말고 복사, X:feature데이터, 이제 fitted_clf : 학습된 estimator가 담겨 있음!!!
                                    #fit(X_train, y_train)해준것 => 결과물 : 학습된 estimator
            self.classifiers_.append(fitted_clf) #그 estimator가 들어감. 즉, for문 끝나면 classifiers_ = [pipe1으로 학습된 estimator, clf2로 학습된 estimator, pipe3로 학습된 estimator]
        return self

    def predict(self, X):
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:  # 'classlabel' 투표

            #  clf.predict 메서드를 사용해 결과를 모읍니다.
            predictions = np.asarray([clf.predict(X)
                                      for clf in self.classifiers_]).T

            maj_vote = np.apply_along_axis(
                                      lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        probas = np.asarray([clf.predict_proba(X)
                             for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        """GridSearch를 위해서 분류기의 매개변수 이름을 반환합니다"""
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:] #여기까지 했을 때 y에는 1,1,1,1,1,..2,2,2,2,2...만 들어가있고 0은 안들어감.
le = LabelEncoder()     #y 안의 값은 모두 숫자인데 레이블인코딩 해주는 이유 : 나중에 bin_count할 때 1부터 시작하면 애매하기 때문에 그냥 바꿔줌
y = le.fit_transform(y) #레이블인코딩 끝난 후 y에는 0,0,0,0,0,.1,1,1,1,1...로 바뀌어 들어가있음

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)
#test_size=0.5, stratify=y로 보면 train, test 데이터는 X row의 반이니까 50의 크기일 것이고, 그 비율은 둘 다 y의 비율을 따라 1:1일 것이다. 

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# estimator 객체 3개 만듦
clf1 = LogisticRegression(solver='liblinear',
                          penalty='l2', 
                          C=0.001,
                          random_state=1)

clf2 = DecisionTreeClassifier(max_depth=1,  #지금 일부러 max_depth를 1로 줘서 너무 얕게 만듦 -> 일부러 약한 분류기를 만들었구나~!
                              criterion='entropy', #get_param()으로 까보면 criterion default값 : gini    //gini, entropy, 불순물지수 전부 비슷한 목적이다. 같다고 생각.
                              random_state=0)

clf3 = KNeighborsClassifier(n_neighbors=1,
                            p=2,
                            metric='minkowski')

pipe1 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf3]])
'''
cf. 파이프라인
pipe1, pipe3 얘네는 나중에 estimator처럼 사용할 것이다. (fit, predict)
estimator는 fit(X_train, y_train)해줘야 하는데, Pipiline은 일단 스케일링 해주고 clf1으로 학습해준다. 
'''

clf_labels = ['Logistic regression', 'Decision tree', 'KNN']

print('10-겹 교차 검증:\n')
for clf, label in zip([pipe1, clf2, pipe3], clf_labels): #1st - pipe1,'Logistic regression', 2nd - clf2,'Decision tree', 
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc') # cross_val_score() : 자동적으로 Stratified K폴드 교차 검증
                    #for문 돌면서 clf에 순차적으로 3개 estimator 전부 들어감    #scoring : 판단 기준
                                                                        
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))




# 다수결 투표 (클래스 레이블 카운트)

mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])

#__init()__에서 뭐 담겨있는지 그냥 확인
print(mv_clf.named_classifiers)
print(mv_clf.vote)
print(mv_clf.weights)


clf_labels += ['Majority voting'] #뒤에 새로 이름 하나 추가
all_clf = [pipe1, clf2, pipe3, mv_clf] #마지막 요소 : 방금 MajorityVoteClassifier로 만든 estimator 객체

for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
    #지금 voting방식을 사용하는 estimator를 train데이터 pipe1, clf2, pipe3, mv_clf로 교차 검증을 수행하고 있는 중!
    
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    '''
    ROC AUC: 0.92 (+/- 0.15) [Logistic regression]
    ROC AUC: 0.87 (+/- 0.18) [Decision tree]
    ROC AUC: 0.85 (+/- 0.13) [KNN]
    ROC AUC: 0.98 (+/- 0.05) [Majority voting]     - 마지막 mv_clf가 가장 좋긴 함. 근데 여기서 중요한 건 그게 아님.
    
    1. 우선 train_test_split
    2. fit()메소드에서 cv=10이므로 10개로 나눠서 1개씩 빼놓고 교차검증
    '''