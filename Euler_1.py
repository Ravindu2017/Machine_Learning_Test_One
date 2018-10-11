s=0
for x in range (3,1000):
    if(x%3==0 or x%5==0):
        s+=x

print(s)

##fibonacci = []
##
##fibonacci[0] = 1

summation = 0


    

##def fibo(n,m):
##    n = 1
##    m = (n + 1) + n
##      
##    if z < 4000000:
##      z = m + n
##        return fibo(n) + n
##    else:
##        return z
##
##print (fibo(1))
##
##m = 3 n + 2 n
##m = n + n-
num_list = []
num_list.append(int(1))
num_list.append(int(2))
num_list.append(int(3))
for i in range(3,4000001):
    num_list[i] += num_list[i-1] + num_list[i-2]

print(num_list[5])
