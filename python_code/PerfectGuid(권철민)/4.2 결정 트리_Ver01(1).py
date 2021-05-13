#!/usr/bin/env python
# coding: utf-8

# ### 결정 트리 모델의 시각화(Decision Tree Visualization)

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# DecisionTree Classifier 생성
dt_clf = DecisionTreeClassifier(random_state=156)

# 붓꽃 데이터를 로딩하고, 학습과 테스트 데이터 셋으로 분리
iris_data = load_iris()
X_train , X_test , y_train , y_test = train_test_split(iris_data.data, iris_data.target,
                                                       test_size=0.2,  random_state=11)

# DecisionTreeClassifer 학습. 
dt_clf.fit(X_train , y_train)


# In[ ]:


from sklearn.tree import export_graphviz

# export_graphviz()의 호출 결과로 out_file로 지정된 tree.dot 파일을 생성함. 
export_graphviz(dt_clf, out_file="C:/jeon/tree.dot", class_names=iris_data.target_names , feature_names = iris_data.feature_names, impurity=True, filled=True)


# In[ ]:


import graphviz

# 위에서 생성된 tree.dot 파일을 Graphviz 읽어서 Jupyter Notebook상에서 시각화 
with open("C:/jeon/tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


# In[ ]:


import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

# feature importance 추출 
print("Feature importances:\n{0}".format(np.round(dt_clf.feature_importances_, 3)))  #dt_clf.feature_importances : 각각의 기여도

# feature별 importance 매핑
'''
for test in zip(iris_data.feature_names , dt_clf.feature_importances_):         #(name, value) 튜플 형태로 묶어서 넘겨줌
    print(test)
'''
for name, value in zip(iris_data.feature_names , dt_clf.feature_importances_):  #이렇게 받는 걸 2개로 지정해주면 튜플로 묶지 않고 각각 들어감
    print('{0} : {1:.3f}'.format(name, value))

# feature importance를 column 별로 시각화 하기 
sns.barplot(x=dt_clf.feature_importances_ , y=iris_data.feature_names)


# ### 결정 트리(Decision TREE) 과적합(Overfitting)

# In[ ]:


from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.title("3 Class values with 2 Features Sample data creation")

# 2차원 시각화를 위해서 feature는 2개, 결정값 클래스는 3가지 유형의 classification 샘플 데이터 생성. 
X_features, y_labels = make_classification(n_features=2, n_redundant=0, n_informative=2,    #(row size - default 100, column size - n_features 2), n_features 2개 중에 y_labels과 상관관계가 있는 피처는 n_informative 2개.
                             n_classes=3, n_clusters_per_class=1,random_state=0)            #label값 n_classes 3개, n_clusters_per_class=1 : y_labels는 '하나의 label데이터 -- 하나의 군집을 이루었다'
    #make_classification(기준) : 기준에 맞는 데이터 리턴

# plot 형태(scatter:점찍는 그래프)로 2개의 feature로 2차원 좌표 시각화, 각 클래스값은 다른 색깔로 표시됨. 
plt.scatter(X_features[:, 0], X_features[:, 1], marker='o', c=y_labels, s=25, cmap='rainbow', edgecolor='k')
            # x축             y축               동그라미로표기   y_label로 클래스 나눔     다른 색깔로 구분


# In[ ]:


import numpy as np

test=[]
test1=[]
test2=[]
test3=[]
test4=[]
test5=[]
test6=[]
test7=[]
test8=[]

# Classifier의 Decision Boundary를 시각화 하는 함수
def visualize_boundary(model, X, y):
    '''
    model : 점 찍은 그래프. 아까 그걸로 decisiontree estimator dt_clf로 학습을 시켜서(ex_너 x 3보다 크니?...계속 트리 타고 내려감) 이 모델 만듦
    X : 피처값 2개(x,y)
    y : 레이블값 3개(0,1,2)-색깔
    '''
    global test
    global test1
    global test2
    global test3
    global test4
    global test5
    global test6
    global test7
    global test8
    
    fig,ax = plt.subplots()
    
    # 학습 데이타 scatter plot(점찍기)으로 나타내기 - 여기까지만 하면 위에서 그려줬던 그냥 점만 찍힌 것과 똑같음
    ax.scatter(X[:, 0], X[:, 1], c=y, s=25, cmap='rainbow', edgecolor='k',
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim_start , xlim_end = ax.get_xlim() #최솟값, 최댓값
    ylim_start , ylim_end = ax.get_ylim()
    test = ax.get_xlim()
    test1 = ax.get_ylim()
    
    # 호출 파라미터로 들어온 training 데이타로 model 학습 . 
    model.fit(X, y)
    # meshgrid 형태인 모든 좌표값으로 예측 수행. 
    test2 = np.linspace(xlim_start,xlim_end, num=200) #xlim_start,ylim_start부터 xlim_end,ylim_end까지 똑같은 step으로 200개 만들어줘라
    test3 = np.linspace(ylim_start,ylim_end, num=200)
    test4, test5 = np.meshgrid(np.linspace(xlim_start,xlim_end, num=200),np.linspace(ylim_start,ylim_end, num=200))
    
    xx, yy = np.meshgrid(np.linspace(xlim_start,xlim_end, num=200),np.linspace(ylim_start,ylim_end, num=200)) 
    '''
    meshgrid(x, y)의 결과
    xx : x 한 row로 두고 그걸 y 크기만큼 아래로 쫙 복사 -> shape : (y크기, x크기)
    yy : y 한 column으로 두고 그걸 x 크기만큼 오른쪽으로 쫙 복사 -> shape : (y크기, x크기)
    '''
    test6 = xx.ravel() #xx를 쫙 한 줄로 이어 붙임 : 0~199인덱스가 반복
    test7 = yy.ravel() #yy를 쫙 한 줄로 이어 붙임 : 0~199인덱스 동일-첫번째y값, 200~399인덱스 동일-두번째y값, ...
    test8 = np.c_[xx.ravel(), yy.ravel()] #xx.ravel();40000크기(1차원데이터) , yy.ravel();40000크기(1차원데이터) 두 개를 묶어서 (40000,2) 만들어짐
                                          #대응되는 좌표(인덱스)의 x,y끼리 예측해서 그 예측값을 reshape
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape) #애초에 model이 앞에서 make_classification에서 100개의 scatter 데이터 그래프로 학습시킨 것. 
                                                                       #그러면 이제는 모~든 좌표! 40000개의 각각의 좌표를 피쳐로 삼아서 학습시킴. - 예측값의 유니크한 값 : 0,1,2
    
    # 자! 이제 모든 좌표에 대해 등고선 그래프 그려보자~
    # contourf() 를 이용하여 class boundary 를 visualization 수행.  //contour:컨투어, 윤곽, 등고선
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap='rainbow', clim=(y.min(), y.max()),
                           zorder=1)


# In[ ]:

#과적합 해결 전 - min_samples_leaf = 1(default)

from sklearn.tree import DecisionTreeClassifier

# 특정한 트리 생성 제약없는 결정 트리의 Decsion Boundary 시각화.
dt_clf = DecisionTreeClassifier().fit(X_features, y_labels)#dt_clf estimator 만들고 그걸로 X_features, y_labels fit까지 다 시켜줌
visualize_boundary(dt_clf, X_features, y_labels)


# In[ ]:

#과적합 해결 - min_samples_leaf = 6으로 설정!!!  //영역 구분이 명확해졌다

# min_samples_leaf=6 으로 트리 생성 조건을 제약한 Decision Boundary 시각화
dt_clf = DecisionTreeClassifier( min_samples_leaf=6).fit(X_features, y_labels)
visualize_boundary(dt_clf, X_features, y_labels)


# ### 결정 트리 실습 - Human Activity Recognition

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# features.txt 파일에는 피처 이름 index와 피처명이 공백으로 분리되어 있음. 이를 DataFrame으로 로드.
feature_name_df = pd.read_csv('C:/jeon/human_activity/features.txt',sep='\s+',      # \s+ : 정규식, 공백 1개 이상 (\x의 의미 : 공백. +의 의미 : 바로앞의 것이 반복되는 것도 해당)
                        header=None,names=['column_index','column_name'])

print(len(feature_name_df)) #컬럼의 개수만 561개나 된다..!

# 피처명 index를 제거하고, 피처명만 리스트 객체로 생성한 뒤 샘플로 10개만 추출
feature_name = feature_name_df.iloc[:, 1].values.tolist()
print('전체 피처명에서 10개만 추출:', feature_name[:10])
feature_name_df.head(20)

'''
X_train_test = pd.read_csv('C:/jeon/human_activity/features.txt',sep='\s+',
                        header=None,names=feature_name)
 
→ ValueError: Duplicate names are not allowed. 안에 중복된 컬럼명(피처명)이 있다는 뜻! 
→ names=feature_name 지정 안해주면 잠깐은 ㄱㅊ 
→ 해결 : 중복처리 !!!
'''

# 우선 중복된 피처명이 얼마나 있는지 알아보기
feature_dup_df = feature_name_df.groupby('column_name').count() #만약 column_name 중에서 중복이 있으면 그것끼리 묶일 것. feature_dup_df의 column_index 피쳐는 중복 개수를 나타낸다. (1 넘으면 중복 있는것)
print(feature_dup_df[feature_dup_df['column_index']>1].count()) #불린인덱싱. feature_dup_df['column_index']>1이 True인 로우만 모아 그 개수 출력  //42개구나~!
feature_dup_df[feature_dup_df['column_index']>1].head()

# ### 수정 버전 01: 날짜 2019.10.27일
# 
# **원본 데이터에 중복된 Feature 명으로 인하여 신규 버전의 Pandas에서 Duplicate name 에러를 발생.**  
# **중복 feature명에 대해서 원본 feature 명에 '_1(또는2)'를 추가로 부여하는 함수인 get_new_feature_name_df() 생성**

# In[5]:

test10 = []
test11 = []
test12 = []
test13 = []

def get_new_feature_name_df(old_feature_name_df):
    global test10
    global test11
    global test12
    global test13
    
    feature_dup_df = pd.DataFrame(data=old_feature_name_df.groupby('column_name').cumcount(), columns=['dup_cnt'])
    test10 = feature_dup_df  #원본데이터랑 크기 same.
    ''' Age     count   cumcount
    0   10      3       0(번째 10)
    1   15      2       0
    2   10              1(번째 10)
    3   10              2(번째 10)
    4   16      1       0
    5   15              1
    '''
    
    feature_dup_df = feature_dup_df.reset_index()
    test11 = feature_dup_df
    
    new_feature_name_df = pd.merge(old_feature_name_df.reset_index(), feature_dup_df, how='outer')
    test12 = old_feature_name_df.reset_index()
    test13 = new_feature_name_df
    ''' merge
            - 특정 key값기준으로 붙이기(how = inner:교집합, outer:합집합) - index 정보(key값)을 기준으로
            - index 삭제 등으로 차이가 있을 때
                inner : 교집합, merge 결과물에서 그 index는 빠짐
                outer : 합집합, merge 결과물에서 그 index에 해당하는 값을 채워서 넣음
        concatenate
            - axis를 줘서 물리적으로 어떻게 붙일 것인지(row방향 or column방향으로 갖다붙이기)
    '''
    
    new_feature_name_df['column_name'] = new_feature_name_df[['column_name', 'dup_cnt']].apply(lambda x : x[0]+'_'+str(x[1]) if x[1] >0 else x[0] ,  axis=1) #중복된 피처명은 뒤에 _1, _2를 붙여서 수정해준다
    #x : new_feature_name_df(test13로 확인 가능) 안에서 'column_name', 'dup_cnt' 두 컬럼만을 모은 DataFrame이 있고, 그 안에서 row 하나 하나씩 담아옴.
    new_feature_name_df = new_feature_name_df.drop(['index'], axis=1)
    return new_feature_name_df
#test_set = get_new_feature_name_df(feature_name_df)

# In[11]:


pd.options.display.max_rows = 999
new_feature_name_df = get_new_feature_name_df(feature_name_df)
new_feature_name_df[new_feature_name_df['dup_cnt'] > 0]


# **아래 get_human_dataset() 함수는 중복된 feature명을 새롭게 수정하는 get_new_feature_name_df() 함수를 반영하여 수정**

# In[12]:


import pandas as pd

def get_human_dataset( ):
    
    # 각 데이터 파일들은 공백으로 분리되어 있으므로 read_csv에서 공백 문자를 sep으로 할당.
    feature_name_df = pd.read_csv('C:/jeon/human_activity/features.txt',sep='\s+',
                        header=None,names=['column_index','column_name'])
    
    # 중복된 feature명을 새롭게 수정하는 get_new_feature_name_df()를 이용하여 새로운 feature명 DataFrame생성. 
    new_feature_name_df = get_new_feature_name_df(feature_name_df)
    
    # DataFrame에 피처명을 컬럼으로 부여하기 위해 리스트 객체로 다시 변환
    feature_name = new_feature_name_df.iloc[:, 1].values.tolist()
    
    # 학습 피처 데이터 셋과 테스트 피처 데이터을 DataFrame으로 로딩. 컬럼명은 feature_name 적용
    X_train = pd.read_csv('C:/jeon/human_activity/train/X_train.txt',sep='\s+', names=feature_name )
    X_test = pd.read_csv('C:/jeon/human_activity/test/X_test.txt',sep='\s+', names=feature_name)
    
    # 학습 레이블과 테스트 레이블 데이터을 DataFrame으로 로딩하고 컬럼명은 action으로 부여
    y_train = pd.read_csv('C:/jeon/human_activity/train/y_train.txt',sep='\s+',header=None,names=['action'])
    y_test = pd.read_csv('C:/jeon/human_activity/test/y_test.txt',sep='\s+',header=None,names=['action'])
    
    # 로드된 학습/테스트용 DataFrame을 모두 반환 
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = get_human_dataset()


# In[13]:


print('## 학습 피처 데이터셋 info()')
print(X_train.info())


# In[14]:


print(y_train['action'].value_counts())


# In[15]:


print(X_train.isna().sum().sum())
#------------- null값이면 True, 아니면 False 담은 DataFrame
#------------------- 컬럼별 True(null값)의 개수
#------------------------- 총 True(null값)의 개수


# In[16]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 예제 반복 시 마다 동일한 예측 결과 도출을 위해 random_state 설정
dt_clf = DecisionTreeClassifier(random_state=156)
dt_clf.fit(X_train , y_train)
pred = dt_clf.predict(X_test)
accuracy = accuracy_score(y_test , pred)
print('결정 트리 예측 정확도: {0:.4f}'.format(accuracy))

# DecisionTreeClassifier의 하이퍼 파라미터 - 그냥 파라미터 default 설정이 뭔지 확인하기 위함
print('DecisionTreeClassifier 기본 하이퍼 파라미터:\n', dt_clf.get_params())
'''
 {'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None,       // 'criterion': 'gini' - 결정트리에서 트리를 뻗어나가는 근간이 되는 불평등지수가 'gini'로 default되어 있구나. 'entropy'넣어도 되겠다~.
  'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 
  'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 
  'min_weight_fraction_leaf': 0.0, 'presort': 'deprecated', 'random_state': 156, 'splitter': 'best'}
'''

# In[17]:


from sklearn.model_selection import GridSearchCV

params = {
    'max_depth' : [ 6, 8 ,10, 12, 16 ,20, 24]
}

grid_cv = GridSearchCV(dt_clf, param_grid=params, scoring='accuracy', cv=5, verbose=1 )
grid_cv.fit(X_train , y_train)
print('GridSearchCV 최고 평균 정확도 수치:{0:.4f}'.format(grid_cv.best_score_))
print('GridSearchCV 최적 하이퍼 파라미터:', grid_cv.best_params_)


# ### 수정 버전 01: 날짜 2019.10.27일  
# 
# **사이킷런 버전이 업그레이드 되면서 아래의 GridSearchCV 객체의 cv_results_에서 mean_train_score는 더이상 제공되지 않습니다.**  
# **기존 코드에서 오류가 발생하시면 아래와 같이 'mean_train_score'를 제거해 주십시요**
# 

# In[19]:


# GridSearchCV객체의 cv_results_ 속성을 DataFrame으로 생성. 
cv_results_df = pd.DataFrame(grid_cv.cv_results_)

# max_depth 파라미터 값과 그때의 테스트(Evaluation)셋, 학습 데이터 셋의 정확도 수치 추출
# 사이킷런 버전이 업그레이드 되면서 아래의 GridSearchCV 객체의 cv_results_에서 mean_train_score는 더이상 제공되지 않습니다
# cv_results_df[['param_max_depth', 'mean_test_score', 'mean_train_score']]

# max_depth 파라미터 값과 그때의 테스트(Evaluation)셋, 학습 데이터 셋의 정확도 수치 추출
cv_results_df[['param_max_depth', 'mean_test_score']]


# In[20]:


max_depths = [ 6, 8 ,10, 12, 16 ,20, 24]
# max_depth 값을 변화 시키면서 그때마다 학습과 테스트 셋에서의 예측 성능 측정
for depth in max_depths:
    dt_clf = DecisionTreeClassifier(max_depth=depth, random_state=156)
    dt_clf.fit(X_train , y_train)
    pred = dt_clf.predict(X_test)
    accuracy = accuracy_score(y_test , pred)
    print('max_depth = {0} 정확도: {1:.4f}'.format(depth , accuracy))


# In[21]:


params = {
    'max_depth' : [ 8 , 12, 16 ,20], 
    'min_samples_split' : [16,24],
}

grid_cv = GridSearchCV(dt_clf, param_grid=params, scoring='accuracy', cv=5, verbose=1 )
grid_cv.fit(X_train , y_train)
print('GridSearchCV 최고 평균 정확도 수치: {0:.4f}'.format(grid_cv.best_score_))
print('GridSearchCV 최적 하이퍼 파라미터:', grid_cv.best_params_)


# In[22]:


best_df_clf = grid_cv.best_estimator_

pred1 = best_df_clf.predict(X_test)
accuracy = accuracy_score(y_test , pred1)
print('결정 트리 예측 정확도:{0:.4f}'.format(accuracy))


# In[23]:


import seaborn as sns

ftr_importances_values = best_df_clf.feature_importances_

# Top 중요도로 정렬을 쉽게 하고, 시본(Seaborn)의 막대그래프로 쉽게 표현하기 위해 Series변환
ftr_importances = pd.Series(ftr_importances_values, index=X_train.columns  )

# 중요도값 순으로 Series를 정렬
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
plt.figure(figsize=(8,6))
plt.title('Feature importances Top 20')
sns.barplot(x=ftr_top20 , y = ftr_top20.index)
plt.show()


# In[ ]:




