'''

input() 실제로 쓰면 x !!
  1. 컴퓨터가 막 돌아가는데, 돌 때 마다 어떤 레지스터(□□□□□... 이렇게 생긴 메모리)를 확인한다. 
  2. 레지스터의 비트들은 모두 0인데 interrupt가 발생한 순간 특정 비트가 1로 변하는 flag가 발생한다. 
  3. '주인님이 지금 내가 돌고 있는 것보다 우선순위가 높은 명령을 내렸구나!' 
  4. 컴퓨터는 인터럽트 테이블을 보고 ex)'마우스가 움직여서 생긴 인터럽트구나!' 확인하고 마우스 제어에 관한 함수로 점프해서 해결하고 돌아온다
기본 순서가 이런데, input()은 인터럽트를 무시하고 모든 일을 멈춰서 잠시 죽은 상황이 된다. 마우스를 움직이는 것도 처리가 안 됨. 
그래서 실제 어플에서는 input() 절대 쓰면 안 되고 인터럽트에 해당하는 '통신'으로 대체함. 

파이썬은 회사에서 마라톤 테스트(어플이 언제 죽는지 가혹한 환경에서 계속 실험) 할 때, 통신연결(프로토콜 구현한 패키지가 있어서 메시지 주고받기 편함)할 때 많이 쓰임
이 때 몇날 몇시 몇분 몇초에 어떤 문제가 생겨서 어떤 부분이 죽었는지 로그 남길 때 f.write 많이 쓰임. 
얘네가 알아서 계속 해주니까 업무 자동화. 

w  write     쓰기 모드
r   read      읽기 모드
a   append  추가 모드
f.readline()   	#수행될 때마다 맨 위부터 한 줄씩 내려가면서 문자열로 묶어 담아줌
f.readlines()     	#파일의 모든 줄을 읽어서 각각의 줄을 str 요소로 갖는 리스트로 담아줌
f.read()          	#파일 전체를 문자열로 묶어 담아줌

f=open('C:/jeon/python_code/test.txt', 'w')     #f : 파일을 담은 객체
for i in range(1, 11):
    data="%d번째 줄입니다.\n"%i
    f.write(data)
f.close()
f=open('C:/jeon/python_code/test.txt', 'r')
while True:
    line=f.readline()
    if not line: break  #line : str. if line:에 역을 취한 것이 if not line:
    print(line)         #"%d번째 줄입니다.\n"의 개행문자에다가 print()에 자동으로 붙어 있는 개행문자가 겹쳐져서 두 번 개행됨
f.close()

with문 : f.close() 빠뜨리는 실수 안하게. with문 벗어나는 순간 f.close() 자동 처리
with open("test.txt", "w") as f:
    f.write("Life is too short, you need python")
'''