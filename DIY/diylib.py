import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def Laplace(Vt,Vb,Vl,Vr,Vg,dt,dimx,dimy,it):
# Set Dimension
    lenX = dimx
    lenY = dimy 

# Colour interpolation and colour map
    colorinterpolation = 20
    colourMap =  plt.cm.coolwarm

# Set meshgrid using numpy lib
    X, Y = np.meshgrid(np.arange(0, lenX), np.arange(0, lenY))
# Set array size and set the interior value with guess V
    V = np.empty((lenX, lenY))
    V.fill(Vg)

# Set Boundary condition
    V[(lenY-1):, :] = Vt
    V[:1, :] = Vb
    V[:, (lenX-1):] = Vr
    V[:, :1] = Vl

# Iteration 
    for iteration in range(0, it):
        for i in range(1, lenX-1, dt):
            for j in range(1, len(Y)-1, dt):
                V[i, j] = 0.25 * (V[i+1][j] + V[i-1][j] + V[i][j+1] + V[i][j-1])
                
# plotting contour
    
    plt.contourf(X, Y, V, colorinterpolation, cmap=colourMap)
    plt.title("Contour for Potential")
    # Set Colorbar
    plt.colorbar(label= 'Potential gradient')
    plt.show()
    print()
    
    return X,Y,V
    