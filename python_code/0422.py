'''

파이썬에서 '배열의 형태로 만들어라~' == '연산 쉽게 ndarray로 만들어라~'
ndarray타입으로 바꾸면 계산도 빨라지고 행렬 연산도 쉽다
type과 shape(==차원, 형태) 중요!

ndarray 배열의 shape변수는 데이터의 크기를 튜플 형태 ex_(행, 열)로 가짐
array1 = np.array([1,2,3])			array1.shape → 1차원데이터 (series) : (3,)      // , 붙이기
array2 = np.array([[1,2,3], [2,3,4]])		array2.shape → 2x3 shape를 가진 2차원데이터 : (2, 3)
array3 = np.array([[1,2,3]])			array3.shape → 1x3 shape를 가진 2차원데이터 : (1, 3)

A=1에서 A와 같이 파이썬에서 모든 변수는 엄밀히 말하면 하나의 객체이다
type(A)는 <class 'int'>, 즉 int라는 객체. 4byte의 공간에 저장되지만, 그것을 위해 사실 더 많은 공간이 필요함
array1 = np.array([1,2,3])에서 type(array1)은 int32 (32bit 공간 안에만 존재하는 숫자 하나로 바뀐 것) 크기도 작아지고 심플해짐

[1 2 3]처럼 공백으로 구분
ndarray명.dtype → 해당 ndarray의 요소의 데이터 타입 리턴
ndarray는 무조건 동일한 데이터타입 가져야 함.      cf. 리스트 안에는 모든 타입 가능
	[int, int, str]을 ndarray로 만들면 [str str str]로 타입캐스팅 		[1, 2, 'test'] → ['1' '2' 'test']
	[int, int, float]을 ndarray로 만들면 [float float float]로 타입캐스팅 	[1, 2, 3.1] → [1. 2. 3.1] 
		#float형.astype(int형) → 소수점 날라감

np.array()♣
인자를 넣어주면 ndarray 데이터타입으로 변환

np.arange()♣
range()와 똑같은데 ndarray 데이터타입으로 만들어질 뿐

ndarray명.reshape()♣
값은 그대로고 shape만 바꿔줌
사용tip 1. 보통 행, 열 둘중에 하나만 정해주고 나머지 -1 넣어줌
	array1 = np.arange(10)
	array2 = array1.reshape(-1,5)
사용tip 2. 2차원 데이터를 1차원으로 reshape해준 뒤 데이터연산 함수에 input → output값을 다시 2차원으로 reshape
	이미지(디스플레이)는 여러 층의 rgb가 겹쳐져있는 개별 픽셀을 모아놓은 데이터임. 영상은 그 이미지가 프레임 단위로 막~~~ 넘어가는 것. 
	이런 이미지 처리도 shape 막~~ 바꿔가면서 함

tolist()♣
리스트로 변환

불린 인덱싱♣
array1d = np.arange(start=1, stop=10)
array3 = array1d[array1d > 5]	#[ ] 안에 array1d > 5 Boolean indexing을 적용
	> [F, F, F, F, F, T, T, T, T]
	> False값은 무시하고 True값에 해당하는 index만 저장 [5, 6, 7, 8]
	> 저장된 index값으로 데이터 조회 array1d[[5, 6, 7, 8]]=[6 7 8 9]
		#ndarray, DataFrame은 이렇게 접근 가능
★
np.sort(ndarray명)		→ 원본 데이터는 냅두고 결괏값을 리턴
ndarray명.sort()		→ 리턴값 없고 원본 데이터 자체를 수정

보통은 inplace라는 파라미터 설정을 통해 리턴해줄지 말지 결정
inplace = False 로 설정 	→ 원본 데이터는 냅두고 결괏값을 리턴 (default설정)
inplace = True 로 설정  	→ 리턴값 없고 원본 데이터 자체를 수정

[::-1] 			→ 내림차순정렬
np.sort(ndarray명, axis=0)	→ 로우 방향, 즉 로우가 증가하는 방향으로 정렬, axis=0
np.sort(ndarray명, axis=1)	→ 컬럼 방향, 즉 컬럼이 증가하는 방향으로 정렬, axis=1
np.argsort()♣		→ 원본 행렬의 인덱스가 정렬되면 어디에 있는지

np.dot(A, B) 		→ A와 B 행렬 내적 (==행렬 곱)
np.transpose(A)		→ A행렬의 전치행렬
cf. 행렬 곱 계산
| 1 2 3 |     | 7  8  |  
| 4 5 6 |  x  | 9 10 |   
               |11 12 |
    ↓            ↓
  2 x 3        3 x 2	     //AxB의 크기 = A의 행 x B의 열
★

Series	   // 컬럼이 1개. 1차원데이터
DataFrame  // 컬럼이 2개 이상. 2차원데이터
DataFrame이라는 형태에서 컬럼이 1개인 것과 Series 인 것은 다르다! (컬럼이 1개라고 무조건 Series는 아님)

인덱싱/슬라이싱	위치기반(iloc)
		명칭기반(loc)
		불린

sort_values() 정렬 	1. by=['컬럼명', '컬럼명', ..]	해당 컬럼을 기준으로 정렬
		2. ascending=True		True:오름차순, False:내림차순
		3. inplace=False		False:원본데이터 안건드림, True:원본데이터 건드림
agg() 		aggregation함수
groupby() 

isna() 		결손 데이터 확인
fillna()		결손 데이터 대체
A.apply(lambda x:...)	똑같이 A는 iterate해야 함. A의 요소에 x가 들어감.

'''