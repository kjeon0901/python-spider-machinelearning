'''
in vs not in
리스트		1 in [1,2,3] == True		1 not in [1,2,3] == False
튜플		'a' in ('a','b','c') == True		'a' not in ('a','b','c') == False
문자열♣		'py' in 'python' == True		'py' not in 'python' == False
딕셔너리		'name' in {'name':'pey'} == True	'name' not in {'name':'pey'} == False

조건문에 아무 것도 넣고 싶지 않다면 pass 쓰거나 아무거나 프린트라도 해야 함

input()에는 임시 데이터 저장 공간인 버퍼가 있음. 이 버퍼 안에 \n(개행, 즉 enter)가 들어가는 순간 처음부터 \n 앞까지의 모든 입력값을 하나의 문자열로 묶어서 리턴해준다. 

무한루프 중단법 : 정지 버튼 누르거나 Ctrl+C

range(부터, 이전까지, 만큼씩증가)

a=[1, 2, 3, 4]
result = [num*3  for num in a  if num%2==0]
           a------  b------------  c--------------
b→c→a 순서로
b. a에서 num을 하나씩 가져오는데
c. 조건문을 만족한 짝수만
a. 3배를 하여 result에 담는다
'''