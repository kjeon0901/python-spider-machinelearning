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

test=[]
test1=[]
test2=[]

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin): #BaseEstimator 상속 -> MajorityVoteClassifier 클래스를 커스터마이징한 estimator처럼 사용 가능 (fit, predict, predict_proba 메소드 보면 알수있듯)
    global test
    global test1
    global test2
    
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
        # 레이블 인코딩에서 fit(A) - A 안의 유니크한 레이블값에 숫자 부여한 테이블 하나 만듦, B=transform(A) - fit로 만든 레이블 테이블을 가져와서 A에 적용시킨 0 1 0 1 2 1... 값을 B에 담음. fit_transform() 로 한번에 해도 됨
        self.lablenc_ = LabelEncoder() #밖에서 우리가 labelencoding 해줬기 때문에 사실 필요 없는데 그냥 해줌
        self.lablenc_.fit(y) #y: 아래에서 붖꽃데이터 test데이터 레이블인코딩 끝내고 더 아래에서 train_test_split해줘서 나온 y_train 데이터. 50개이고 비율은 0:1=1:1 
        self.classes_ = self.lablenc_.classes_ #레이블 인코딩 된 아이의 유니크한 값이 들어감. 여기선 0, 1 두개가 들어감
        self.classifiers_ = []
        for clf in self.classifiers: #클래스 객체 만들 때 받아온 estimator 3개 담긴 리스트. dtype:list, 각각의 요소 dtype:estimator
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y)) #clone():원본말고 복사, X:feature데이터, 이제 fitted_clf : 학습된 estimator가 담겨 있음!!!
                                    #fit(X_train, y_train)해준것 => 결과물 : 학습된 estimator
                                    #근데 교차검증이기 때문에 45개의 데이터만 요기서 함. 이제 나머지 5개의 test데이터로 predict 해봐야 함.
                                    #근데 처음에 교차검증 scoring='roc_auc'(x축이 FPR로, threshold값의 변화에 따라 그래프 그려짐)를 해주었기 때문에, 확률값이 필요하다. predict가 아니라, predict_proba를 해줘야 함. 그래서 자동으로 그렇게 실행되므로 22222222222222222222222가 출력된다. 
                                    #roc_auc - threshold값 변화 - predict_proba에서 레이블값 나누는 비율 변화
            self.classifiers_.append(fitted_clf) #그 estimator가 들어감. 즉, for문 끝나면 classifiers_ = [pipe1으로 학습된 estimator, clf2로 학습된 estimator, pipe3로 학습된 estimator]
        return self

    def predict(self, X): #교차검증이기 때문에 X는 5개의 데이터(shape:5x2, cv=10이었기 때문에)
        print('111111111111111111111111111111')
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:  # 'classlabel' 투표 - default가 여기임. 

            #  clf.predict 메서드를 사용해 결과를 모읍니다.
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T #.T 붙이면 전치행렬! 행렬 뒤집기. 우리가 원하는 건 하나의 estimator예측이 하나의 column에 들어가길 원한다. 
            test1.append(predictions.T) #이미 한번 전치행렬 해둬서 AT 됐으니까 (AT)T 해서 다시 A 볼 수 있음. 
            test2.append(predictions) #3개의 estimator가 5개의 데이터를 예측한 예측값을 5x3형태로 담음
            
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.weights)), axis=1, arr=predictions)
            #보통 df.apply(lambda x: ~~) 하면 df에서 row 하나씩 x에 들어가는데, 여기서는 특정짓지 않고 np라고 해주었다. 대신 뒤에 arr = predictions라고 해줘서 predictions에 적용하는 것을 알 수 있다. 
            '''
            이게 Hard Voting과 연결되는 이유!!! ★★★
            3개의 estimator가 각각의 데이터에 예측한 값에서 다수결로 최종 결정한다. 
            '''
        maj_vote = self.lablenc_.inverse_transform(maj_vote) 
        #레이블 인코딩해서 조금 달라진 값을 다시 원래대로 0, 1 -> 1, 2. 근데 지금 보면 이미 밖에서 레이블 인코딩 끝난 뒤에 train_test_split해주었다. 그리고 이 위에서 혹시 빠진 경우가 있을까봐 한번 더 레이블인코딩 해주었다. (여기선 별로 의미 없었지만)
        #그래서 지금 inverse_transform해봤자, 위에 두 번째로 수행한 레이블 인코딩만 다시 원래대로 돌려놓게 된다. 그래봤자 돌려놓은 상태도 그대로 0, 1 -> 0, 1인 걸!!!
        #여기선 별 의미 없지만, 다른 케이스들을 생각해보면 (레이블인코딩 한 번만 수행했고, 그게 split 이후라면) 이 코드는 유의미해진다!
        return maj_vote

    def predict_proba(self, X): #교차검증이기 때문에 X는 5개의 test 데이터(shape:5x2, cv=10이었기 때문에)
        print('222222222222222222222222222222')
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_]) #probas : 3x5x2 3차원 데이터, 각 estimator별로 predict_proba한 결과 확률값 5x2데이터가 3번 담기니까. 
        test=probas #3차원 데이터인 것 확인~
        avg_proba = np.average(probas, axis=0, weights=self.weights) #3차원이기 때문에 axis=0,1,2까지 가능. axis=0이면 3x5x2에서 3에 해당. 3개에서 같은 위치에 있는 애들끼리 평균 내고, 결괏값은 5x2 shape.
        '''
        이게 Soft Voting과 연결되는 이유!!! ★★★
        3개의 classifier들의 각각의 피쳐 데이터에 해당하는 확률을 레이블값별로 낸 평균을 구함. 
        여기서 최고인 레이블값으로 최종 class값 결정하기만 하면 Soft Voting!
        '''
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
le = LabelEncoder()     #y 안의 값은 모두 숫자인데 레이블인코딩 해주는 이유 : 나중에 bincount할 때 1부터 시작하면 애매하기 때문에 그냥 바꿔줌
                        #bincount([0,0,1,3,3,3]):[2 1 0 3], argmax(bincount([0,0,1,3,3,3])):3(최곳값 있는 인덱스)
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

pipe1 = Pipeline([['sc', StandardScaler()], #그냥 clf1 이름이 pipe1로 바꼈구나~라고 생각
                  ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()], #그냥 clf3 이름이 pipe3으로 바꼈구나~라고 생각
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

mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3]) #리스트 안에 estimator도 요소로 넣을 수 있다. classifiers라는 파라미터 설정으로 이것 넣어줌

#__init()__에서 뭐 담겨있는지 그냥 확인
print(mv_clf.named_classifiers)
print(mv_clf.vote)
print(mv_clf.weights)


clf_labels += ['Majority voting'] #뒤에 새로 이름 하나 추가
all_clf = [pipe1, clf2, pipe3, mv_clf] #마지막 요소 : 방금 MajorityVoteClassifier로 만든 estimator 객체

for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='accuracy') #얘가 accuracy로 바뀌면 predict_proba 말고 predict가 실행되어 11111111111111111111이 출력된다. 
    #지금 voting방식을 사용하는 estimator pipe1, clf2, pipe3, mv_clf를 train데이터로 교차 검증을 수행하고 있는 중!
    #50개의 데이터가 10조각으로 fold되어 45개, 5개로 나누어 교차검증
    
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    '''
    ROC AUC: 0.92 (+/- 0.15) [Logistic regression]
    ROC AUC: 0.87 (+/- 0.18) [Decision tree]
    ROC AUC: 0.85 (+/- 0.13) [KNN]
    ROC AUC: 0.98 (+/- 0.05) [Majority voting]     - 마지막 mv_clf가 가장 좋긴 함. 근데 여기서 중요한 건 그게 아님.
    
    1. 우선 train_test_split
    2. fit()메소드에서 cv=10이므로 10개로 나눠서 1개씩 빼놓고 교차검증
    '''