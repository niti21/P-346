#!/usr/bin/env python
# coding: utf-8

# In[3]:


import math
import matplotlib.pyplot as plt
#Defining functions
#def f(q):
    #return (math.log(q/2)-math.sin((5*q)/2))
#def derivF(q):
   # return ((1/(q/2)*2.3026)-((5/2)*math.cos((5*q)/2)))
#def deriv2F(q):
    #y = 12*(q**2)- 6*(q) -14 
    #return y


#To find derivative
def df(c1):
    l = len(c1)
    for i in range (0,l):
        c1[i]= c1[i]*i
        cf = c1
    print ('1st derivative:',cf)
    return cf

#to find 2nd derivative
def d2f(c2):
    for i in range (1,l+2):
        c2[i]= c2[i]*(i-1)
    return c2

#Deflation
def deflate(c,a):
    root = x
    k = len(c)
    for i in range (0,k-1):
        c[i+1] = c[i+1]+ root*c[i]  
       #print (c) 
    return c

def deflation(c,a):
    if len(c) != 1:
        c[1] = c[1] + c[0]*a
        for i in range(2,len(c)):
            c[i] = c[i] + a*c[i - 1]
        c.pop()
    else:
        print("cannot deflate")
    return c

#Recurssion deflation
def def2(c):
    for a in range(k-1,2,-1):
        if c[a]==0:
            x = c[a-1]
        d = deflate(c,x)
        print(x)
    return d

#bracketing
def bracket(f,a,b):
    t=0
    bt = 0.1
    while (f(a)*f(b)>=0.0):
        t=t+1
        if (abs(f(a)) < abs(f(b))):
            a=a-bt*(b-a)
        if (abs(f(a)) > abs(f(b))):
            b= b+bt*(b-a)
       #print()
        
   #print()
   #print('Bracketing steps: ',t)
    return a,b,t

#Bisection Method
def bisection(f, a, b, tol):
    a,b,s= bracket(f, a, b)
    ct = 0
    root_i = []
    abs_error = []
    max_iter = 200   # maximum iterations allowed
    for i in range(max_iter):
        c = (a+b)/2
        prod = f(a) * f(c)
        if prod < 0:
            b = c
        elif prod > 0:
            a = c
       #print(ct,' : ',c)
        ct+=1
        root_i.append(c)
        error = abs(root_i[i] - root_i[i-1])
        abs_error.append(error)

        if abs(a-b)<tol:

           #print(c)
    
           #print(abs_error)
           #print()
          # print('No. of iterations: ', ct)
            return c,ct

#Regula Falsi Method
def regulaFalsi(f, a, b, tol):
    a,b,s= bracket(f, a, b)
    ct = 0
    root_i = []
    abs_error = []
    max_iter = 200
    c = a  # initial guess for the root
    for i in range(max_iter):
        c_prev = c
        c = b - ((b-a)*f(b))/(f(b) - f(a))
        if f(a) * f(c) < 0:
            b = c
        elif f(a) * f(c) > 0:
            a = c
       #print(ct,' : ',c)
        ct+=1
        root_i.append(c)
       #print(root_i)
        error = abs(root_i[i] - root_i[i-1])
        abs_error.append(error)

        if abs(c - c_prev) < tol:
           #print(c)
           #print( 'No. of iterations:',ct)
            return c,ct 
    
def compare(f,a,b,tol):
    a1,b1,s= bracket(f, a, b)
    a2,b2,s= bracket(f, a, b)
    ct = 0
    ct1 = 0
    root_ri=[]
    root_bi= []
    abs_error = []
    
    max_iter = 25  # maximum iterations allowed
    c1= a2  # initial guess for the root
    for i in range(max_iter):
        c = (a1+b1)/2
        prod = f(a1) * f(c)
        if prod < 0:
            b1 = c
            ct+=1
        elif prod > 0:
            a1 = c
            ct+=1
        root_bi.append(c)
        if abs(a1 - b1) < tol:
            break
    
    
    for i in range(max_iter):
        
        c_prev = c1
        c1= b2 - ((b2-a2)*f(b2))/(f(b2) - f(a2))
        if f(a2) * f(c1) < 0:
            b2 = c1
        
        elif f(a2) * f(c1) > 0:
            a2 = c1
        ct1+=1
        root_ri.append(c1)
        
        if abs(c1- c_prev) < tol:
            break
    bl=len(root_bi)
    rl = len(root_ri)
   #print(rl)
    for i in range(rl,bl):
        root_ri.append('-')
        
    for i in range(bl):
        print(i+1,' : ',root_bi[i],' : ',root_ri[i])
        
def copy_list(c):
    cn = []
    for i in range(0,len(c)):
        a = c[i]
        cn.append(a)
    return cn

def derivative(z):
   
    for i in range(0,len(z)):
        z[i] = z[i]*(len(z)-i-1)
    z.pop()
   
    return z

def fn(c,z):
    sum1 =0
    for i in range(0,len(c)):
        sum1 = sum1 + c[i]*(z**(len(c)-i-1))
    return sum1
    
#Laguerre Method for 1 root
def laguerreMethod(x0, n, e):
    xk = x0
    t=0
    while abs(f(xk)) > e:
        G = derivF(xk) / f(xk)
        H = (G ** 2) - deriv2F(xk) / f(xk)
        #n= len(c)-1
        root = math.sqrt((n - 1) * (n * H - G ** 2))
        d = max(abs(G + root), abs(G - root))
        a = n / d
        xk = xk - a
        t=t+1
    print("Root: ", xk)
    print("No. of iterations:", t)
    return xk
#Laguerre Method for all roots
def laguerre(c,z,degree):
    t = 0
    tol= 0.000001
    k1 = copy_list(c)
    k2 = copy_list(c)
    k2 = derivative(k2)
   
    while t < degree:
        if abs(fn(c,z))<tol:
           #print(fn(c,z))
            print('Root: ',z)
        else:
            k1 = derivative(k1)
            k2 = derivative(k2)
            # print(c)
            # print(k1)
            # print(k2)
            count = 0
            while abs(fn(c,z))>tol:
                f = z
               
                sum1 = 0
                for i in range(0,len(k1)):
                    sum1 = sum1 + k1[i]*(z**(len(k1)-i-1))
                G = sum1/fn(c,z)
               
                sum2 = 0
                for i in range(0,len(k2)):
                    sum2 = sum2 + k2[i]*(z**(len(k2)-1-i))
                H = G**2 - (sum2/fn(c,z))
               
                n = len(c)-1
                if G < 0:
                    a = n/(G - ((n-1)*(n*H - G**2))**0.5)
                else:
                    a = n/(G + ((n-1)*(n*H - G**2))**0.5)
                z = z - a
                count = count + 1
        
        print('Root: ',z)
        #print(count,'no. of iterations: ')
        c = deflation(c,z)
       #print(c)
        k1 = copy_list(c)
       #print(k1)
        k2 = copy_list(c)
        k2 = derivative(k2)
       #print(k2)
        t=t+1

#calling function
#c = [1,2,3,4]
#z = 0
#laguerre_2(c,z,4)

#Newton raphson method
def newtonRaphson(f,df,q,tol):
    t=0
    h = f(q) / df(q)
    while abs(h) >= tol:
        h = f(q)/df(q)
        q = q - h
        t=t+1
    return q,t

# Partial pivoting
def pivot(A,B):
    n = len(A)
    for i in range(n-1):
        if A[i][i] == 0:
            for j in range(i+1,n):
                if A[j][i] > A[i][i]:
                    A[i],A[j] = A[j],A[i]
                    B[i],B[j] = B[j],B[i]
    return A,B

def gauss_jordan(A,B):
    n = len(A)
    
    #partial pivoting
    A,B=pivot(A, B)             
    for i in range(n):
        p = A[i][i]
        #Making diagonal terms 1
        for j in range(i,n):
            A[i][j] = A[i][j]/p
        B[i] = B[i]/p 

        for k in range(n):              
            #Making Column zero except diagonal
            if (k == i) or (A[k][i] == 0):
                continue
            else:
                f = A[k][i]
                for l in range(i,n):
                    A[k][l] = A[k][l] - f*A[i][l]
                B[k] = B[k] - f*B[i]
   #print('Gauss jordan solution:')
    
    return B

def augmentMat(z,x):
    for i in range(0,len(z)):
        z[i].append(x[i])
       #print(z[i])
    return z
def ftrac(r,WW):
        h=0
        for i in range(0,len(WW)):
            h=h+(WW[i]*(r**i))
        return h
def polySF(X,Y,degree): 
    V= []
    for i in range(0,degree):
        a = []
        for j in range(0,degree):
            a.append(0)
        V.append(a)
    z = []
    c = []
    for j in range(0,2*len(V)-1):
        sum1= 0
        sum2 = 0
        for i in range(0,len(X)):
            sum1 =sum1+(X[i]**j)
            sum2 = sum2 + Y[i]*(X[i]**(j))
        z.append(sum1)
        c.append(sum2)
    for i in range(degree,len(z)):
        c.pop()
    for i in range(0,len(V)):
        for j in range(0,len(V)):
            V[i][j] = z[i+j]

    WW=[]
   #print(c)
    WW=gauss_jordan(V,c)
    arr_f=[]
    for i in range(0,len(X)):
        arr_f.append(ftrac(X[i],WW))
    
    plt.plot(X,Y,'bo')
    plt.plot(X,arr_f,'yo-')
   
    return c


