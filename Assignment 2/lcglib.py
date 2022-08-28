#!/usr/bin/env python
# coding: utf-8

# In[34]:


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
        


# In[ ]:




