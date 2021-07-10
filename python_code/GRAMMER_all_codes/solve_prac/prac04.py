#■■■■■■■■■■■■■04장 연습문제■■■■■■■■■■■■■
#1
def is_odd(number):
    if number%2:
        return True
    else:
        return False
number=int(input("자연수 입력:"))
print("입력값은 홀수가 %s"%is_odd(number))

#2
def avg_numbers(*args):
    result=0
    for i in args:
        result +=i
    return result/len(args)
print(avg_numbers(1, 2))
print(avg_numbers(1,2,3,4,5))

#3
input1=int(input("첫번째 숫자를 입력하세요:"))
input2=int(input("두번째 숫자를 입력하세요:"))
total=input1+input2
print("두 수의 합은 %s입니다"%total)

#5
f1=open("C:/jeon/python_code/test.txt", 'w')
f1.write("Life is too short\n")
f1.close()
f2=open("C:/jeon/python_code/test.txt", 'r')
print(f2.read())
f2.close()

#6
user_input=input("저장할 내용을 입력하세요:")
f=open('C:/jeon/python_code/test.txt', 'a')
f.write(user_input)
f.write("\n")
f.close()

#7
f=open('C:/jeon/python_code/test.txt', 'r')
body=f.read()
f.close()
body=body.replace('python', 'java')
f=open('C:/jeon/python_code/test.txt', 'w')
f.write(body)
f.close()