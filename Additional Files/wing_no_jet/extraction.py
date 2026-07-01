import math
import numpy as np

from helper import returnCamberLine
def getvalue():
    cl=[]
    aoa1=[]
    for aoa in range(0,10):
        aoa = aoa* math.pi / 180
        aoa1.append(aoa)
        # Normalized camber line coordinates
        xCamber, yCamber = returnCamberLine("Airfoils/NACA4412.csv") # Can be changed to any airfoil file (like NACA23015.csv)
        numberOfPoints = len(xCamber)

        # We solve the vortex sheet equation through linear algebra A * gamma = velTerms
        camberLineSlope = []
        anot = 0.0
        an = []

        # Calculate camber line slope at each point
        for a in  range (0, numberOfPoints-1):
            camberLineSlope.append((yCamber[a+1] - yCamber[a]) / (xCamber[a+1] - xCamber[a]))
        camberLineSlope.append((yCamber[a+1] - yCamber[a]) / (xCamber[a+1] - xCamber[a]))

        # For integration along the camber line, we convert from 
        # the cartesian x to the theta variable
        theta = []
        for a in range (0, numberOfPoints):
            theta.append(np.arccos(1 - 2 * xCamber[a]))
        theta.append(theta[a] + 0.001)

        # For the A_0 term:
        for a in range(0, numberOfPoints):
            anot = anot + camberLineSlope[a] * (theta[a+1] - theta[a]) 
        anot = aoa - 1 / math.pi * anot

        # For the A_n terms:
        for N in range(0, numberOfPoints):
            an.append(0)
            for n in range(0, numberOfPoints):
                an[N] = an[N] + camberLineSlope[n] * math.cos(N*theta[n]) * (theta[n+1] - theta[n])
            an[N] = 2 / math.pi * an[N] 

        # Finally, we calculate the coefficient of lift

        cl.append(math.pi * (2 * anot + an[1]))

    return cl,aoa1