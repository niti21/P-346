#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import math
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
        


# In[4]:


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
    print('Gauss jordan solution: ')
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


# In[5]:
#Defining functions
def fc(q):
    y = (q**4) - (q**3)-7*(q**2) + q + 6
    return y
def derivFc(q):
    y = 4*(q**3)- 3*(q**2) -14*(q) + 1
    return y
def deriv2Fc(q):
    y = 12*(q**2)- 6*(q) -14 
    return y


#To find derivative
def dfc(c1):
    l = len(c1)
    for i in range (0,l):
        c1[i]= c1[i]*i
        cf = c1
    print ('1st derivative:',cf)
    return cf

#to find 2nd derivative
def d2fc(c2):
    for i in range (1,l+2):
        c2[i]= c2[i]*(i-1)
    return c2

#Deflation
def deflate(c,x):
    root = x
    k = len(c)
    for i in range (0,k-1):
        c[i+1] = c[i+1]+ root*c[i]  
        print (c) 
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
def bracket(a,b,func):
    t=0
    while (f(a)*f(b)>=0):
        t=t+1
        if (abs(f(a)) < abs(f(b))):
            a=a-0.1
        if (abs(f(a)) > abs(f(b))):
            b= b+0.1
        print()
        call = func
    print()
    print('Steps: ',t)
    return call

#Bisection Method
def bisection(a,b,tol):
    if (f(a)*f(b)>=0):
        print('wrong interval')
    c = a
    print ("%10.2E" %a, "%10.2E" %f(a))
    print ("%10.2E" %b, "%10.2E" %f(b))
    while ((b-a)>= tol):
        for i in range(15):
            c=(a+b)/2
            if (f(c)==0):
                break
            if (f(c)*f(a)<0):
                b=c
            else:
                a=c
    print ("%10.2E" %c, "%10.2E" %(f(b)))
    return c

#Regula Falsi Method
def regulaFalsi(a,b,tol):
    if f(a) * f(b) >= 0:
        print("Wrong interval")
        return -1
    o=0 
    c = a 
    print (a, f(a))
    print (b, f(b)) 
    for i in range(n):
        #point touching x axis
        c = (b - ((b - a) * f(b))/ (f(b) - f(a))) 
        # Find root
        if f(c) == tol:
            break
        elif f(c) * f(a) < 0:
            b = c
        else:
            a = c
    print("Roots : " , '%10.2E' %c, 'at: ', '%10.2E' %f(c) )
    
#Laguerre Method
def laguerreMethod(x0, n, e):
    xk = x0
    t=0
    while abs(f(xk)) > e:
        G = derivF(xk) / f(xk)
        H = (G ** 2) - deriv2F(xk) / f(xk)
        root = math.sqrt((n - 1) * (n * H - G ** 2))
        d = max(abs(G + root), abs(G - root))
        a = n / d
        xk = xk - a
        t=t+1
    print("Root: ", xk)
    print("No. of iterations:", t)
    return xk

#Alternative
def deflation(c,a):
    if len(c) != 1:
        c[1] = c[1] + c[0]*a
        for i in range(2,len(c)):
            c[i] = c[i] + a*c[i - 1]
        c.pop()
    else:
        print("cannot deflate")
    return c
#########################################
def copy_list(c):
    cn = []
    for i in range(0,len(c)):
        a = c[i]
        cn.append(a)
    return cn
#######################################
def derivative(z):
   
    for i in range(0,len(z)):
        z[i] = z[i]*(len(z)-i-1)
    z.pop()
   
    return z

#########################################
def fn(c,z):
    sum1 =0
    for i in range(0,len(c)):
        sum1 = sum1 + c[i]*(z**(len(c)-i-1))
    return sum1
###########################################
def laguerre_2(c,z,degree):
    t = 0
    f = 7
     
    k1 = copy_list(c)
    k2 = copy_list(c)
    k2 = derivative(k2)
   
    while t < degree:
        if abs(fn(c,z))<0.0000001:
            print(fn(c,z))
            print(z, "is a root")
        else:
            k1 = derivative(k1)
            k2 = derivative(k2)
            # print(c)
            # print(k1)
            # print(k2)
            count = 0
            while abs(fn(c,z))>0.0000001:
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
        print(z,"is a root")
        # print(count,'is the number of iterations used to get',z)
        c = deflation(c,z)
        k1 = copy_list(c)
        k2 = copy_list(c)
        k2 = derivative(k2)
        t=t+1
#calling function
#c = [1, -1, -7, 1, 6]
#z = 0
#laguerre_2(c,z,4)


#Newton raphson method
def newtonRaphson(q,tol):
    t=0
    h = f(q) / derivF(q)
    while abs(h) >= tol:
        h = f(q)/derivF(q)
        q = q - h
        t=t+1
     
    print("Root: ", q)
    print("No. of iterations:", t)
    return q

#Data Fit

#Interpolation
def interpolate(a,b,x):
    s = 0
    l = len(a)
    for i in range(l):
        p = 1
        for k in range(l):
            if k != i:
                p *= ((x - a[k])/(a[i]-a[k]))
        s+=p*b[i]
    print (x,':', s)
    print(l,a,b)
    return s
    
#Least Sq fit
def augmentMat(z,x):
    for i in range(0,len(z)):
        z[i].append(x[i])
        print(z[i])
    return z

def polySF(X,Y,degree):
    plt.plot((X),Y)
    plt.show()  
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
    c.pop()
    c.pop()
    for i in range(0,len(V)):
        for j in range(0,len(V)):
            V[i][j] = z[i+j]
    return V,c
#Calling fn
#import matplotlib.pyplot as plt
#V,c =least_sq_fit(X,Y,3)
#augmentMat(V,c)
#gauss_jordan(V,c)

####################################################################
#Pearson R
def persons_ratio(x,y):
    #for sigma =1
    n=len(x)
    sx,sy,sxy,sx2,sy2=0,0,0,0,0
    for i in range(n):
        sx+=x[i]
        sy+=y[i]
        sxy+=x[i]*y[i]
        sx2+=x[i]**2
        sy2+=y[i]**2
    R=(n*sxy - sx*sy)/(((n*sx2 - sx*2)**0.5)*((n*sy2 - sy*2)**0.5))
    return R

#Linear Fit
def linear_fit(X,Y):
    # plt.scatter((X),Y)
    # plt.show()  
    sum_x = 0
    sum_y = 0
    sum_x2 = 0
    sum_xy = 0
    sum_y2 = 0
    N = len(X)
    for i in range(0,len(X)):
        sum_x = sum_x + (X[i]/sigma[i])
        sum_x2 = sum_x2 + ((X[i]**2)/sigma[i])
        sum_y = sum_y + (Y[i]/sigma[i])
        sum_xy = sum_xy + ((X[i]*Y[i])/sigma[i])
        sum_y2 = sum_y2 + ((Y[i]**2)/sigma[i])
    a1 = (N*sum_xy-sum_x*sum_y)/(N*sum_x2-(sum_x**2))
    a2 = (sum_y - a1*sum_x)/(N)
   
    delta_x = N*sum_x2-(sum_x**2)
    delta_y = N*sum_y2-(sum_y**2)
    r = ((N*sum_xy - sum_x*sum_y)**2)/(delta_x*delta_y)
    return a1,a2,r

#X = [4050,4360,4920,5460,5770,6230,6910]
#Y = [4060,4380,4910,5460,5760,6150,6870]
#sigma = [1,1,1,1,1,1,1]
#########################################################

#Integration

#Midpoint integration
def midpoint_method(a,b,N,f):
    h = (abs(a - b))/N
    x = []
    y = []
    # print(h)
    for i in range(1,N+1):
        x.append((2*a + (2*i-1)*h)/2)
       
        y.append(f(x[i-1]))
    # print(y)
    # print(x)
    sum1 = 0
    for j in range(0,len(y)):
        sum1 = sum1 + y[j]*h
    print(round(sum1,9))
 
#trapeziodal integration
def trapezoidal(a,b,N,f):
    h = (abs(a - b))/N
    x = []
    y = []
    # print(h)
   
    for i in range(0,N+1):
        x.append((a + i*h))
        y.append(f(x[i]))
    sum1 = 0
    for i in range(1,len(x)):
        sum1 = sum1 + (h/2)*(y[i-1]+y[i])
    print(round(sum1,9))
    
#Simpsons
def Simpsons( l, u, n ):
 
    # Calculating the value of h
    h = ( u - l )/n
 
    # List for storing value of x and f(x)
    x = list()
    fx = list()
    # Calculating values of x and f(x)
    i = 0
    while i<= n:
        x.append(l + i * h)
        fx.append(f(x[i]))
        i += 1
 
    # Calculating result
    res = 0
    i = 0
    while i<= n:
        if i == 0 or i == n:
            res+= fx[i]
        elif i % 2 != 0:
            res+= 4 * fx[i]
        else:
            res+= 2 * fx[i]
        i+= 1
    res = round((res * (h / 3)),9)
    return res

#Finding N
#def f(x):
#    return 1/x

def N_simp(l,u,t):
    N = ((1/180*t)*24)**(1/4)
    return N
def N_mid(l,u,t):
    N = ((1/24*t)*2)**(1/2)
    return N
def N_trap(l,u,t):
    N = ((1/12*t)*2)**(1/2)
    return N

