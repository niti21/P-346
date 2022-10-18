#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plot
a = 1103515245
c = 12345
m = 32768.
x = 10
def lcg(a,c,m,x):
    y = ((a*x + c) % m)/m
    return y
def lcgshow(a,c,m,x):
    rnum = []
    for i in range(0,500):
        y = (a*x + c) % m
        rnum.append(y)
        x=y
        i= i+1  
        plt=plot.scatter(i, y) 
    return rnum
def lcgplot(a,c,m,x):
    rnum = []
    for i in range(0,500):
        y = (a*x + c) % m
        rnum.append(y)
        x=y
        i= i+1  
        plt=plot.scatter(i, y) 
    return plt

def display(x):
    print(lcgshow(1103515245, 12345, 32768, x))
    lcgplot(1103515245, 12345, 32768, x)
        


# In[2]:


def PrintMatrix(a, n):
    for i in range(n):
        print(a[i])
        
#To check for tolerance
def matTol(x,y,tol):
    count=0
    if len(x)==len(y):
        for i in range(0,len(x)):
            if (abs(x[i]-y[i]))/abs(y[i]) < tol:
                count =count+1
            else:
                return False
    if count==len(x):
        return True

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

#Swapping individual rows
def swap(A,i,j):
    A[i],A[j]=A[j],A[i]
    return A
    
#Checks if matrix is symmetric
def symmetric(A):                                            
    i=0
    cnt=0
    while i<len(A):
        j=0
        while j<len(A):
            if A[i][j]==A[j][i]:
                cnt+=1    
            j+=1
        i+=1
    if cnt==len(A)*len(A):
        print('Matrix is Symmetric!')
        return True
    else:
        print('Not Symmetric!')
        return False

#Transpose of a square matrix
def transpose(a):
    i=0
    while i<len(a):
        j=i+1
        while j<len(a):
            a[i][j],a[j][i] = a[j][i],a[i][j]
            j+=1
        i+=1
    return a

#For a diagonally dominant matrix.(to find the maximum element in a column and swap it with the diagonal element.)
def DiagDom(A,X):
    i=0
    n=len(A)
    for i in range(n):               
        maxima=A[0][i]              
        for j in range(n):
            if A[j][i]>maxima:
                maxima=A[j][i]
                swap(A, i, j)
                swap(X, i, j)
                if A[j][j]==0:
                    swap(A, i, j)
                    swap(X, i, j)
    return A,X

def lud(A):
    #Define matrices
    lt = [[0 for x in range(len(A))]
             for y in range(len(A))]
    ut = [[0 for x in range(len(A))]         
             for y in range(len(A))]
 
    for i in range(len(A)):                 
        #lower and upper matrix decomposition
        for k in range(i, len(A)):
            sum1 = 0
            for j in range(i):
                sum1 += (lt[i][j] * ut[j][k])       
            ut[i][k] = round((A[i][k] - sum1),4)  
            
        #Making diagonal terms 1 for solving equation.
        for k in range(i, len(A)):
            if (i == k):
                lt[i][i] = 1                                         
            else: 
                sum1 = 0                                                    
                for j in range(i):
                    sum1 += (lt[k][j] * ut[j][i])
                lt[k][i] = round(((A[k][i] - sum1)/ut[i][i]),4)
    print()
    print('Lower triangle:')
    print (lt)
    print()
    print('Upper triangle:')
    print (ut)          
    return lt,ut

#To decomposes A into Lower and Upper which are transpose of each other if A is symmetric
def cholesky(A):
    #Check for symmetric matrix
    if symmetric(A):                                         
        i=0
        while i <len(A):
            j=0
            temp=0
            while j<i:
                temp+=round((A[j][i]*A[j][i]),4)
                j+=1
            A[i][i]=round(((A[i][i]-temp)**(0.5)),4)                
            j=i+1
            while j<len(A):
                k=0
                temp=0
                #Recurrence relations
                while k<i:
                    temp+= round((A[i][k]*A[k][j]),4)
                    k+=1
                A[j][i]=round(((A[j][i]-temp)/A[i][i] ),4)          
                A[i][j]=A[j][i]
                j+=1
            i+=1
        i=0
        while i <len(A):                                   
            j=i+1
            while j<len(A):
                A[i][j]=0
                j+=1
            i+=1
        
        print()
        print("Cholesky Decomposed matrix: ")
        print(A)
        At = transpose(A)
        print()
        print('Transpose of the decomposed matrix:')
        print(At)
        return A, At
    
def forwardsub(A,B):
    i=0
    Y=[]
    for k in range(len(A)):
        Y.append(0)
    while i<len(A):
        j=0
        temp=0
        while j<i:
            temp+=A[i][j]*Y[j]
            j+=1
        Y[i]=round(((B[i]-temp)/A[i][i]),4)
        i+=1
    print()
    return Y

def backwardsub(A,Y):
    i=len(A)-1
    X=[]
    for l in range(len(A)):
        X.append(0)
    while i>=0:
        j=i+1
        temp=0
        while j<len(A):
            temp+=A[i][j]*X[j]
            j+=1
        X[i]=round(((Y[i]-temp)/A[i][i]),4)
        i-=1
    print()
    return X

#function for gauss-jordan elimination
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
    return B
#For guess matrix
def gsm(a,b,x):
    for j in range(0, len(a)):               
        d = b[j]
        
        for i in range(0, len(a)):     
            if(j != i):
                d = d - a[j][i] * x[i]
        x[j] = round((d / a[j][j]),4)
    return x

#For gauss seidel (n = no. of iterations)
def GaussSeidel(a,x,b,n,tol): 
    y=[]
    for i in range(0,len(x)):
        y.append(x[i])
    
    for i in range(1,n):
        t=0
        for t in range(0,len(x)):
            y[t] = x[t]
    
        x = gsm(a,b,x)
    
        print(i,':',x)
    
        z = matTol(x,y,tol)
        if z == True:
            print()
            print('Gauss seidel result: ',x, '; no. of iterations: ',i)
            break
        else:
            continue

#Function for jacobi
def jacobi(a,b,it,tol):
    print(" Jacobi Calculations: \n")
    
    x=[1]*len(a)
    x1=[1]*len(a)
    
    
    for i in range(0,it):
        t=0
        for t in range(0,len(x)):
            x[t] = x1[t]
        
        x1 = gsm(a,b,x1)
    
        print(i+1,".",x1)
    
        z = matTol(x1,x,tol)
        if z == True:
            print('Jacobi result: \n',x1,"\n after",i+1,"iterations.")
            break
        else:
            continue

