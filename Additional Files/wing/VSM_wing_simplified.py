import math
import numpy as np

from helper import returnCamberLine

# Angle of Attack in radians
aoa = 8 * math.pi / 180

# Normalized camber line coordinates
xCamber, yCamber = returnCamberLine("Airfoils/NACA2412.csv")
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

# This presents the simplified method to calculate the coefficient of lift directly
cl = 0
for a in range(0, numberOfPoints):
    cl += camberLineSlope[a] * (math.cos(theta[a]) - 1) * (theta[a+1] - theta[a])

# Finally, we calculate the final coefficient of lift
cl = 2 * math.pi * (aoa + cl / math.pi)
print(cl)