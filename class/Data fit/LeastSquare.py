import matplotlib.pyplot as plt
import math

def least_squares(data1,data2):
   sd = 1
   n = len(data1)
   Sx, Sy, Sxx, Sxy, S, R, Syy = 0, 0, 0, 0, 0, 0, 0

   for i in range(len(data1)):
       Sx += data1[i]/(sd)**2
       Sy += data2[i]/(sd)**2
       Sxy += (data1[i]*data2[i])/(sd)**2
       Sxx += (data1[i]*data1[i])/(sd)**2
       Syy += (data2[i]*data2[i])/(sd)**2
       S += 1/sd**2

   d = S*Sxx - Sx**2
   R = math.sqrt((n*Sxy - Sx*Sy)**2/((n*Sxx - Sx**2)*(n*Syy - Sy**2)))

   return (Sxx*Sy - Sx*Sxy)/d,(Sxy*S - Sx*Sy)/d, R

def f1(x):
    return 1/x

def f2(x):
    return x*math.cos(x)

def polyfc(coeff,x):
    fc = 0
    for i in range(len(coeff)):
        fc += (coeff[i])*(x**(i))
    return coeff, fc

def dpolyfc(coeff,x):
    for i in range(len(coeff)):
        if i == 0:
            coeff[i] = 0
        coeff[i] = i*coeff[i]
    coeff.pop(0)
    df = polyfc(coeff,x)
    return df


def MidInt(a,b,N):
    h = (b-a)/N
    M = 0
    for i in range(N+1):
        M += h*f2(a+(((2*i+1)/2))*h)

    return M

def trapezInt(a,b,N):
    h = (b-a)/N
    M = 0
    for i in range(N+1):
        if i == 0 or i == N:
            M += (h/2)*f2(a+(i)*h)
        else:
            M += (h)*f2(a+(i)*h)
    return M

def SimpsonInt(a,b,N):
    h = (b-a)/N
    M = f2(a) + f2(b)
    for i in range(1,N):
        if i%2 == 0:
            M += 2*f2(a+(i)*h)
        else:
            M += 4*f2(a+(i)*h)

    M *= h/3
    return M
