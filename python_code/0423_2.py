##################### titanic 살짝 건드려보기 #############################

import numpy as np
import pandas as pd

titanic_df = pd.read_csv('C:/jeon/titanic_train.csv')

#1. 결측치(비어있는 값), 데이터타입(하나의 컬럼은 하나의 데이터 타입으로 이루어져야 함) 분석
print(titanic_df.info())
test=titanic_df.describe()
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
value_counts = titanic_df['Age'].value_counts()  #Age에 대한 Series(1개의 컬럼)의 요소 별 그에 해당하는 인원

titanic_df['Age_0']=0    #새로운 컬럼 추가하고 0으로 모두 채우기
titanic_df.head(3)


#2. DataFrame과 리스트, 딕셔너리, 넘파이 ndarray의 상호 변환 - 사용하기 쉽게~

#3. DataFrame의 컬럼 데이터 세트 생성, 수정
titanic_df['Age_by_10'] = titanic_df['Age']*10    #컬럼의 연산의 결과는 각각의 개별 요소끼리의 연산 결과를 담은 컬럼. 하나의 Series.
titanic_df['Family_No'] = titanic_df['SibSp']+titanic_df['Parch']+1
titanic_df.head(3)
titanic_df['Age_by_10'] = titanic_df['Age']+100
titanic_df.head(3) #바뀜

#4. DataFrame 데이터 삭제 - axis=0인지 axis=1인지 잘 써줘야 함!!!
titanic_drop_df = titanic_df.drop('Age_0', axis=1 )     #y축에서 'Age_0' 컬럼 삭제
titanic_drop_df = titanic_df.drop(8, axis=0 )           #x축에서 8인덱스 로우 삭제 
titanic_drop_df.head(3)
drop_result = titanic_df.drop(['Age_0', 'Age_by_10', 'Family_No'], axis=1, inplace=False)   #inplace=False : 리턴하고 원본데이터는 그대로
print(' inplace=True 로 drop 후 반환된 값:',drop_result)
titanic_df.head(3) #그대로


#5. Index 객체 추출
indexes = titanic_df.index
print(indexes)

print('Index 객체 array값:\n',indexes.values) # .values를 통해 Index 객체를 실제 값 ndarray 데이터타입으로 변환 
indexes_value = indexes.values
'''
cf. 한번 만들어진 DataFrame 및 Series의 Index객체는 개별 row를 구분하는 "유니크한 값"(중복X)이기 때문에 수정 불가능
'''
print(type(indexes.values))
print(indexes.values.shape)
print(indexes[:5].values)
print(indexes.values[:5])
print(indexes[6])

series_fair = titanic_df['Fare']   #DataFrame의 컬럼 하나가 Series 타입으로 넘어감
print('Fair Series max 값:', series_fair.max())
print('Fair Series sum 값:', series_fair.sum())
print('sum() Fair Series:', sum(series_fair))
print('Fair Series + 3:\n',(series_fair + 3).head(3) )  #컬럼의 연산 : 각 개별 요소의 연산~~

titanic_reset_df = titanic_drop_df.reset_index(inplace=False)   #.reset_index()로 새 인덱스 생성, 기존의 인덱스는 새로운 컬럼으로 들어감                 
value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)
print('value_counts 객체 변수 타입:',type(value_counts))
new_value_counts = value_counts.reset_index(inplace=False)      #titanic_df['Age'].value_counts()는 나이대로 정렬되는 게 아니라 ,
print(new_value_counts)                                         #유니크한 나이에 해당하는 요소의 개수(해당 나이의 인원은 몇명인가)에 따라 내림차순 정렬됨. 
print('new_value_counts 객체 변수 타입:',type(new_value_counts))


#6. 데이터 셀렉팅 및 슬라이싱 - 위치기반 iloc, 명칭기반 loc
data = {'Name': ['Chulmin', 'Eunkyung','Jinwoong','Soobeom'],
        'Year': [2011, 2016, 2015, 2015],
        'Gender': ['Male', 'Female', 'Male', 'Male']}
data_df = pd.DataFrame(data, index=['one','two','three','four'])
data_df

print(data_df.iloc[0, 0])           #iloc select
print(data_df.loc['one', 'Name'])   #loc select
print('위치기반 iloc slicing\n', data_df.iloc[0:1, 0])              #iloc slice
''' one    Chulmin'''  #one 은 index 출력된 것. 인덱스는 위치에 포함 안하니까~~!! 사실상 Chulmin만 출력됨.
print('명칭기반 loc slicing\n', data_df.loc['one':'two', 'Name'])   #loc slice
''' one     Chulmin
    two    Eunkyung''' #loc는 명칭기반이므로 특별하게 슬라이싱에서 [a:b]이면 b까지 포함!!

titanic_boolean = titanic_df[titanic_df['Age'] > 60] #boolean slice
print(type(titanic_boolean))
titanic_df[titanic_df['Age'] > 60][['Name','Age']].head(3) #'Name' 단일컬럼만 보려면 'Name'만 []에 넣어주면 되지만,
                                                            #'Name', 'Age'처럼 여러 컬럼 보려면 그걸 묶은 "리스트"를 넣어줘야 한다! ['Name', 'Age']리스트를 []에 넣어줌
print(type((titanic_df['Age'] > 60) & (titanic_df['Pclass']==1) & (titanic_df['Sex']=='female')))
print(titanic_df[ (titanic_df['Age'] > 60) & (titanic_df['Pclass']==1) & (titanic_df['Sex']=='female')])
'''cond1 = titanic_df['Age'] > 60
cond2 = titanic_df['Pclass']==1
cond3 = titanic_df['Sex']=='female'
titanic_df[ cond1 & cond2 & cond3]    얘를 줄여 쓴 것'''


#7. 정렬, Aggregation(집합) 함수-agg()♣, GroupBy 적용-groupby()♣
titanic_sorted = titanic_df.sort_values(by=['Name'])    #'Name' 컬럼을 기준으로 정렬해주세요~
titanic_sorted.head(3)
titanic_sorted = titanic_df.sort_values(by=['Pclass', 'Name'], ascending=[False, True]) #처음으로 들어온 'Pclass'로 정렬한 뒤, 'Pclass'는 건들지 않고, 같은 'Pclass' 안에서 'Name'으로 정리
titanic_sorted.head(3)                                                                  #'Pclass'는 내림차순, 'Name'은 오름차순으로 정렬

'''앞으로 aggregation함수를 쓸 때 무조건 agg('agg함수명')로 묶어서 ~!!'''
print(titanic_df.agg('count')) #각 컬럼에서 null값이 아닌(데이터가 애초에 없으니까 자동 제외) value만 count
print(titanic_df[['Age', 'Fare']].agg('mean')) #'Age', 'Fare'에서 null값이 아닌(데이터가 애초에 없으니까 자동 제외) value들의 평균

titanic_groupby = titanic_df.groupby('Pclass')
print(type(titanic_groupby))
print(titanic_groupby) #그냥 객체가 나와버렸다... 얘를 Aggregation함수와 함게 써야 의미가 있다!
titanic_groupby = titanic_df.groupby('Pclass').agg('count') #'Pclass'의 유니크한 값 1, 2, 3을 이용해 각 칼럼별로 null값이 아닌 value만 count
print(type(titanic_groupby))
print(titanic_groupby)  #Cabin이 처음에 titanic_df.info()로 봤을 때 null값이 엄청 많았는데, titanic_groupby로 확인하니까 Pclass가 1인 경우에는 그렇게 많지 않았다..!
                        #그와중에 Cabin이 2, 3인 경우는 많았다. 즉, 1등급이 아닌 2, 3등급 선실에서 묵은 사람들의 명부는 엄청 많이 누락되어 있구나~!
titanic_groupby = titanic_df.groupby('Pclass')[['PassengerId', 'Survived']].agg('count') #'Pclass'의 유니크한 값 1, 2, 3으로 카테고리 만들고, 12개의 컬럼 중에서 요 두개만 뽑아서 나타낼게요~

titanic_df.groupby('Pclass')['Age'].agg([max, min]) #'Pclass'의 유니크한 값 1, 2, 3을 카테고리로 하는 'Age'컬럼만을 보는데, 각각의 'Age'의 max, min 구한다                                                  
agg_format={'Age':'max', 'SibSp':'sum', 'Fare':'mean'} #딕셔너리 - key:컬럼명, value:적용시킬 aggregation 함수
titanic_df.groupby('Pclass').agg(agg_format) 


#8. 결손 데이터 처리 - 1. 해당 row 날려버리기 2. 해당 column 날려버리기 3. 평균값으로 채우기 4. 0으로 채우기
    #isna() : 결손 데이터 확인
print(titanic_df.isna()) #titanic_df의 전체 데이터가 각각 null인지 모두 확인
print(titanic_df.isna().sum())  #titanic_df.isna()의 결과인 DataFrame에서 각 컬럼의 sum값을 출력. 
                                #titanic_df.isna()의 value는 모두 boolean값(null이면 True==1, 아니면 False==0)이므로 True값, 즉 null의 개수를 세준다. 
    #fillna() : 결손 데이터 대체
titanic_df['Cabin'] = titanic_df['Cabin'].fillna('C000') #titanic_df['Cabin']에서 null을 'C000'로 대체
print(titanic_df['Cabin'].isna().sum())
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean()) #titanic_df['Age']에서 null을 평균으로 대체
titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S') #titanic_df['Embarked']에서 null을 'S'으로 대체
print(titanic_df.isna().sum())


#9. A.apply(lambda x:...) 식으로 데이터 가공    → A는 iterate, A의 요소가 x에 들어감.
lambda_square = lambda x : x ** 2    # a**b == a^b
print('3의 제곱은:',lambda_square(3))
titanic_df['Name_len']= titanic_df['Name'].apply(lambda x : len(x)) #'Name_len' 컬럼 추가해서 'Name'컬럼의 단일요소(str)의 길이를 넣어줌
titanic_df['Child_Adult'] = titanic_df['Age'].apply(lambda x : 'Child' if x <=15 else 'Adult') #'Child_Adult' 컬럼 추가해서 'Age'컬럼의 단일요소들이 15 이하면 'Child'를, 아니면 'Adult'를 단일요소로 넣어줌
titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : 'Child' if x<=15 else ('Adult' if x <= 60 else 'Elderly')) #'Age_cat' 컬럼 추가해서 'Child_Adult' 컬럼보다 더 세분화해서 'Elderly'까지 단일요소로 넣어줌
titanic_df['Age_cat'].value_counts()                                                                                  #lambda x: A if a else(B if b else C) 형태 : a면 A, a가 아닌데 b면 B, 그것도 아니면 C                                                                                                                      
def get_category(age):  # 나이에 따라 세분화된 분류를 수행하는 함수 생성
    cat = ''
    if age <= 5: cat = 'Baby'
    elif age <= 12: cat = 'Child'
    elif age <= 18: cat = 'Teenager'
    elif age <= 25: cat = 'Student'
    elif age <= 35: cat = 'Young Adult'
    elif age <= 60: cat = 'Adult'
    else : cat = 'Elderly'
    return cat
titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : get_category(x)) #get_category(X)는 입력값으로 ‘Age’ 컬럼 값을 받아서 해당하는 cat 반환
titanic_df['Age_cat'].value_counts()                                        #lambda에서 if else문 쓰는 것보다 이게 더 나음
