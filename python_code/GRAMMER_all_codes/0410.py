import csv 

f=open('C:/jeon/gihu.csv')
main_data = csv.reader(f) 
print(main_data) 

#필요한 데이터만 main_pt에 모으기
temp=[]                   
for row in main_data:   
    temp.append(row)  
f.close()      

#temp의 핵심 데이터(12번째 row부터 끝 row까지)만 담기
main_pt=temp[8:]           

#내 생일 당일의 최고기온 구하기
for idx, row in enumerate(main_pt):
    if row and row[0] and row[4]:
        if row[0]=='2000-09-01':    #'2000-09-01' in row[0]도 여기서는 답이 맞긴 함. 
            print(row[4])
