import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #데이터값 분포 보기 위함
%matplotlib inline
import seaborn as sns
sns.set() # setting seaborn default for plots

train=pd.read_csv("C:/jeon/titanic_train.csv")
test=pd.read_csv("C:/jeon/titanic_test.csv")

print(train.head()) #데이터 불러오기 잘 됐는지 확인. dataframe 중 앞의 5개의 열을 출력


#10가지 특성
'''
Survived	Survival	                0 = No, 1 = Yes
Pclass	    Ticket class	            1 = 1st, 2 = 2nd, 3 = 3rd
Sex	        Sex	 
Age	        Age in years	 
SibSp	    # of siblings / spouses aboard the Titanic	 
Parch	    # of parents / children aboard the Titanic	 
Ticket	    Ticket number	 
Fare	    Passenger fare	 
Cabin	    Cabin number	 
Embarked	Port of Embarkation	        C = Cherbourg, Q = Queenstown, S = Southampton
'''




#데이터 기본 분석 - 결측치(비어있는 값), 데이터타입(하나의 컬럼은 하나의 데이터 타입으로 이루어져야 함) 분석
print('train data shape: ', train.shape)
print('test data shape: ', test.shape)
print('----------[train infomation]----------')
print(train.info())
print('----------[test infomation]----------')
print(test.info())
'''
Train 데이터는 891개의 행이 있습니다. Age에 177개의 결측지, Cabin에 687개 결측치 그리고 Embarked에는 2개의 결측치가 있습니다.
Age : 결측치가 많지않고 나이에 따라 생존 여부와 상관 있을 것으로 예상되어 데이터 채워 넣어야 함
Cabin : 객실 번호가 생존 여부와 관련 있을 수 있으나 결측치가 너무 많기때문에 제거하는 것이 나을 것으로 보임
Embarked : 2개의 결측치가 있으므로 어느 값으로 채워도 문제 없어 보임
'''
print(train.describe())
'''
count   non-null인 데이터 개수
mean    평균
std     표준편차
min     최솟값
25%     25% 위치의 값
50%     50% 위치의 값
75%     75% 위치의 값
max     최댓값
'''



#데이터 그래프로 분석 - Survived 컬럼과 다른 컬럼들 간의 상관관계 분석 : 10가지 특성이 생존에 미치는 영향

# 1 - Pie Chart 만드는 함수
def pie_chart(feature):
    #y축 기능 가져오기
    feature_ratio = train[feature].value_counts(sort=False)        #feature에 대한 Series(1개의 컬럼)의 유니크한 value값들에 해당하는 인원
    feature_size = feature_ratio.size
    feature_index = feature_ratio.index
    survived = train[train['Survived']==1][feature].value_counts() #생존자 중 해당 특성을 가진 인원
    dead = train[train['Survived']==0][feature].value_counts()     #사망자 중 해당 특성을 가진 인원
    
    #배에 있는 전체 인원을 해당 특성으로 나눈 비율
    plt.plot(aspect='auto')
    plt.pie(feature_ratio, labels=feature_index, autopct='%1.1f%%')
    plt.title(feature+'\'s ratio in total')
    plt.show()
    
    #특성을 기준으로 나뉜 종류별 생존자/사망자 비율
    for i, index in enumerate(feature_index):
        plt.subplot(1, feature_size+1, i+1, aspect='equal')
        plt.pie([survived[index], dead[index]], labels=['Survived', 'Dead'], autopct='%1.1f%%')
        plt.title(str(index)+'\'s ratio')
    plt.show()
    
pie_chart('Sex') #Sex에 대한 Pie Chart
'''남성이 여성보다 배에 많이 탔으며, 남성보다 여성의 생존 비율이 높다는 것을 알 수가 있다.'''
pie_chart('Pclass') #Pclass(사회경제적 지위)에 대한 Pie Chart
'''Pclass가 3인 사람들의 수가 가장 많았으며, Pclass가 높을수록(숫자가 작을수록; 사회경제적 지위가 높을수록) 생존 비율이 높다는 것을 알 수 있다.'''
pie_chart('Embarked') #승차한곳(배 정박 위치)에 대한 Pie Chart
'''Southampton에서 선착한 사람이 가장 많았으며, Cherbourg에서 탄 사람 중에서는 생존한 사람의 비율이 높았고, 나머지 두 선착장에서 탄 사람들은 생존한 사람보다 그렇지 못한 사람이 조금 더 많았다.'''


# 2 - Bar Chart 만드는 함수
def bar_chart(feature):
    survived=train[train['Survived']==1][feature].value_counts()    #생존자 중 해당 특성을 가진 인원
    dead=train[train['Survived']==0][feature].value_counts()        #사망자 중 해당 특성을 가진 인원
    df = pd.DataFrame([survived, dead]) #판다스 행렬 만들기
    df.index=['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10, 5))

bar_chart('SibSp') #SipSp(배우자나 형제 자매 명 수의 총 합)에 대한 Bar Chart
'''2명 이상의 형제나 배우자와 함께 배에 탔을 경우 생존한 사람의 비율이 컸다는 것을 볼 수 있고, 그렇지 않을 경우에는 생존한 사람의 비율이 적었다는 것을 볼 수 있다.'''
bar_chart('Parch') #Parch(부모 자식 명 수의 총 합)에 대한 Bar Chart
'''SibSp와 비슷하게 2명 이상의 부모나 자식과 함께 배에 탔을 때는 조금 더 생존했지만, 그렇지 않을 경우에는 생존한 사람의 비율이 적었다.'''





#데이터 전처리 및 특성 추출 - 학습시킬 특성을 골라 학습

train_and_test=[train, test]
'''
우리가 선택할 특성은 Name, Sex, Embarked, Age, SibSp, Parch, Fare, Pclass
Ticket과 Cabin에 대한 의미는 아직 찾지 못했으므로 데이터 세트에서 제외

데이터 전처리 과정에서 train과 test 데이터를 같은 방법으로 한 번에 처리를 해야하므로 먼저 두 개의 데이터를 합쳐보도록하자.
'''

# 1 - Name Feature : Mr, Mrs 등의 타이틀 추출해서 새롭게 만든 Title 컬럼에 넣어주기
for dataset in train_and_test:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\. ')  #공백으로 시작하고, .으로 끝나는 문자열을 추출(Mr. Mrs.에서 . 앞의 Mr Mrs만 떼어냄)
print(train.head(5))    #train_and_test 안의 train, test는 원래의 train, test와 같은 주소를 가짐

pd.crosstab(train['Title'], train['Sex']) #추출한 Title을 가진 사람이 몇 명이 존재하는지 성별과 함께 표현
for dataset in train_and_test: 
    #Capt, Col, Countess, Don, Dr, Lady, ... 등 흔하지 않은 title은 Other로 대체
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dona', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir'], 'Other')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss') #중복되는 표현은 통일
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')  #중복되는 표현은 통일
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss') #중복되는 표현은 통일
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

for dataset in train_and_test:
    dataset['Title'] = dataset['Title'].astype(str) #추출한 Title데이터를 학습하기 알맞게 str로 타입캐스팅
    

# 2 - Sex Feature : 이미 male, female로 나뉘어 있으므로 str로 타입캐스팅만 해주기
for dataset in train_and_test:
    dataset['Sex'] = dataset['Sex'].astype(str)
    
    
# 3 - Embarked Feature : 
train.Embarked.value_counts(dropna=False)
#train.info()했을 때 NaN값 (결측치) 있었음




