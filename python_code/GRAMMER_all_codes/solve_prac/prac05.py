#■■■■■■■■■■■■■05장 연습문제■■■■■■■■■■■■■
#1
class Calculator:
    def __init__(self):
        self.value=0
    def add(self, val):
        self.value+=val
class UpgradeCalculator(Calculator):
    def minus(self, val):
        self.value-=val
cal = UpgradeCalculator()
cal.add(10)
cal.minus(7)
print(cal.value)

#2
class MaxLimitCalculator(Calculator):
    def __init__(self):     #해줄 필요 없지만 초기값 설정 이외에도 쓰이므로 습관적으로 써주자!
        self.value=0
    def add(self, val):
        self.value+=val
        if self.value>100:
            self.value=100
cal=MaxLimitCalculator()
cal.add(50)
cal.add(60)
print(cal.value)

#3
all([1, 2, abs(-3)-3])  #False
chr(ord('a'))=='a'      #True

#4 filter와 lambda를 사용하여 리스트 [1,-2,3,-5,8,-3]에서 음수를 모두 제거해보자
print(list(filter(lambda x:x>0, [1,-2,3,-5,8,-3])))
#4___풀어쓰면
A=[1,-2,3,-5,8,-3]
B=filter(lambda x:x>0, A)
print(list(B))

#5
print(int(0xea))
print(int('0xea', 16))  #지금 '0xea'는 16진수로 표기되어 있는데, 얘를 10진수로 바꿔주세요!

#6 map과 lambda를 사용하여 리스트 [1,2,3,4]의 각 요솟값에 3이 곱해진 리스트 [3,6,9,12]를 만들어보자
print(list(map(lambda x:x*3, [1,2,3,4])))
#6___풀어쓰면?
A=[1,2,3,4]
B=map(lambda x:x*3, [1,2,3,4])
print(list(B))

#7
A=[-8,2,7,5,-3,5,0,1]
print(max(A), min(A))

#8
print(round(17/3, 4))