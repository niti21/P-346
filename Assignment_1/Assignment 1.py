#!/usr/bin/env python
# coding: utf-8

# ## Q1 Calculate the sum of first N odd numbers and factorial of N (using do-while or for loop).

# In[1]:


# Calculate the sum of first N odd numbers. 
# using for loop
def oddsum(N):
    s=0 
    for i in range(1, 2*N+1): 
        if i%2!=0: 
            s+=i
    return s 


# factorial of N.  
def fact(N):
    f = 1
    for i in range(1, N):  
        f = f*(i+1)
        i+=i
    return f
def display():
    print('Sum of first ' ,N,' odd numbers = ', oddsum(N))
    print()
    print('Factorial of ' ,N,' = ', fact(N))

#[taking N=12]
N=12 
display()


# ## Q2 Calculate the sum of N terms of an AP, GP and HP series for common difference 1.5 and common ratio 0.5.

# In[3]:


# Calculate the sum of N terms of an AP series for common difference 1.5.

def sumAP(a,N,d):
    S = [a]
    s = a
    for i in range(1, N): 
        b = a + d
        s = s + b
        S.append(b) 
        a=b
   
    return S, s

# Calculate the sum of N terms of a GP series of common ratio 0.5.
    
def sumGP(a,N,r):    
    S = [a]
    s = a
    for i in range(1, N): 
        b = a*r
        S.append(b) 
        s = s + b
        a=b
    
    return S,s

#Calculate the sum of N terms of a HP series for common difference 1.5.

def sumHP(c,N,d):
    S = [1/c]
    s = 1/c
    for i in range(1, N): 
        k = c + d
        s = round((s + 1/k),2)
        S.append(round((1/k),2)) 
        c=k
    return S,s
       
def display():
    print ('The AP series and its sum is \n', sumAP(a,N,d)) 
    print()
    print ('The GP series and its sum is \n', sumGP(b,N,r)) 
    print()
    print ('The HP series and its sum is \n', sumHP(c,N,d)) 
  

 #[taking N=12]

N = 12
a = 7
b = 640
c = 2
d = 1.5    
r = 0.5
display()


# ## Q3 Calculate the sum of the series given below accurate up to 4 place in decimal, where n = 1, 2, . . .. Plot the sum versus n.

# In[1]:


# Calculate the sum of the series given below accurate up to 4 place in decimal, where n = 1, 2, . . .. Plot the sum versus n.

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as p

def accurate4():
    a = 1
    P = 0
    Q = 0
    n = 1
    #the loop goes on until the previous sum and the next sum has a difference less than  that of 0.00005
    while abs(Q) >= 0:
        P = Q
        a = ((-1/2)**(n))*(-1)
        Q = Q + a
        p.scatter(Q,n)
        if abs(Q - P) < 0.00005:
            break
        n = n + 1
    return Q

def display():
    print("Sum of the series accurate to 4 decimal places is ", accurate4() )
    print()

display()


# ## Q4 Find AB, D · C and BC.

# In[ ]:


#Calling input files with the matrices
def input():
    with open('4a.txt', 'r') as p:
        A = [[float(num) for num in line.split(',')] for line in p ]
    print("Matrix A is:",A)

    with open('4b.txt', 'r') as q:
        B = [[float(num) for num in line.split(',')] for line in q ]
    print("Matrix B is:",B)

    with open('4c.txt', 'r') as r:
        C = [[float(num) for num in line.split(',')] for line in r ]
    print("Matrix C is:",C)

    with open('4d.txt', 'r') as o:
        D = [[float(num) for num in line.split(',')] for line in o ]
    print("Matrix D is:",D)
    print()

# Matrix multiplication
def Mmultiply(x,y):
    r1 = [[0 for i in range(0,len(y[0]))] for j in range(len(y))]  
    # iterate through rows of A  
    for i in range(len(x)):  
       for j in range(len(y[0])):  
           for k in range(len(y)):  
               r1[i][j] += x[i][k] * y[k][j] 
    
    print ('Product = ',r1)
        

#Dot product 
def dotproduct(x,y):
    r2 = [[0 for i in range(0,len(y[0]))] for j in range(len(x[0]))]
    for i in range(len(x)):   
        r2[0][0] = r2[0][0] + x[i][0]*y[i][0]  
    print ('Dot Product = ',r2)

  


# In[ ]:


input()
#for AB
Mmultiply(A,B)
#for D.C
dotproduct(D,C)
#for BC
Mmultiply(B,C)  


# ## Q5 Define your own class / structure myComplex and calculate the sum, product and modulus of (3 − 2i) and (1 + 2i).

# In[45]:


import math
from math import sqrt

class myComplex(object):
    def __init__(self,r,i):
        self.r = int(r)
        self.i = int(i)

#defining the functions for complex numbers
#Addition
def complexadd(a,b):
    x = a.r + b.r
    y = a.i + b.i
    print("Sum =",x,"+",y,'i')

#Multiplication
def complexmultiply(a,b):
    x = (a.r * b.r) - (a.i * b.i)
    y = (a.r * b.i) + (a.i * b.r)
    print("Product = ",x,"+",y,'i')
    
#Modulus
def complexmod(a):
    x = (a.r * a.r)
    y = (a.i * a.i)
    mod = math.sqrt((x*x) + (y*y))
    mod = round(mod,4)
    print("Modulus = |",mod,"|")

a = myComplex(3,-2)
b = myComplex(1,2)

complexadd(a,b)
complexmultiply(a,b)
complexmod(a)
complexmod(b)

