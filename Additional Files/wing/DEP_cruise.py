"""
Assumptions-
- The airfoil is thin
- The angle of attack of the airfoil must be <<1 radian
- This jet is further approximated to be a vortex sheet

Differences from the paper implementation-
- The camber of the airfoil can be a be non symmetrical airfoil

Variables to change - 
- Angle of attack in the aoaVarFunc
- Chord in the chordVarFunc
- camber (airfoil)

"""

import csv
import matplotlib.pyplot as plt
import math
import numpy as np

#returns the un normalised camber points of the airfoil
def returnCamberLine(camberLineFileLocation):
    with open(camberLineFileLocation, mode='r', encoding='utf-8') as file:
        # Create a csv.reader object to iterate over lines in the CSV file
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        xCamber = []
        yCamber = []

        x_index = header.index('X')
        y_index = header.index('Y')

        for row in csv_reader:
            xCamber.append(float(row[x_index]))
            yCamber.append(float(row[y_index]))
    return(xCamber, yCamber)

#this function is responsible for mapping the jet of the blown flap
def jetModel(lastCamberLineSlope, lastCamberLine, aoa):
    #we define the function of the jet and attach it to the back of the flap
    #we first try to find the x position where the angle between the fsv and the jet is the same as the flap and fsv
    jetLength = 100 #This defines how long the jet is. It should be infinite, but eventually it merges with the fsv and its contribution reduces
    jetResolution = 1 #This defines the step size of the jet
    jetCoeffA = 2.05    #These are two coefficients that are responsible for the jet
    jetCoeffm = 0.28
    jetCoeffr = math.pow((rhoJet * velocityJet * velocityJet)/ (rho * fsv * fsv), 1/2)
    jetStartPoint = math.pow(-math.tan(math.atan(lastCamberLineSlope) - aoa) / (jetCoeffA * math.pow(jetCoeffr * heightJet, 1-jetCoeffm) * jetCoeffm), 1 / (jetCoeffm - 1))
    jetX = []
    jetY = []
    intXStartPoint = 0
    while(True):
        if intXStartPoint < jetStartPoint:
            intXStartPoint += 1
        else:
            break

    for a in range (intXStartPoint, intXStartPoint + jetResolution*jetLength):
        jetX.append(a / jetResolution)
        jetY.append(-jetCoeffA * math.pow(jetCoeffr * heightJet, 1-jetCoeffm) * math.pow(jetX[a - intXStartPoint - 1] , jetCoeffm))

    jetXfinal = []
    jetYfinal = []
    for a in range(0, len(jetX)):
        jetXfinal.append(jetX[a] * math.cos(aoa) + jetY[a] * math.sin(aoa))
        jetYfinal.append(-jetX[a] * math.sin(aoa) + jetY[a] * math.cos(aoa))
    jetXfinaldash = []
    jetYfinaldash = []
    jetYfinaldash = [number - jetYfinal[0] + lastCamberLine for number in jetYfinal]
    jetXfinaldash = [number - jetXfinal[0] + airfoilLen for number in jetXfinal]
    return(jetXfinaldash, jetYfinaldash, len(jetXfinal))

def twoDimentionalAirfoil(aoa):
    xCamber, yCamber = returnCamberLine(r"Airfoils\NACA4412.csv") #This should be the path to where the airfoil is stored
    #We scale the values such that we can use it in our equations (normalization). We need all the values of the camber to be between 0 and 1. 
    yCamber = [number * normalizationVal for number in yCamber]
    xCamber = [number * normalizationVal for number in xCamber] 
    numberOfPoints = len(xCamber)

    for a in range(0, numberOfPoints):
        
        if xCamber[a] > (1 - percFlap/100):
            flapPointLim = a
            break
        elif percFlap == 0:
            flapPointLim = -1
    
    if flapPointLim == -1: #This is for the case without flaps
        pass

    else:   #This is for the case with flaps
        for a in range(flapPointLim, numberOfPoints):
            yCamber[a] = yCamber[a] - (xCamber[a] - xCamber[flapPointLim])* math.tan(flapAngle)
    
    #we solve the vortex sheet equation through linear algebra A * gamma = velTerms

    camberLineSlope = []
    anot = 0.0
    an = []

    for a in  range (0, numberOfPoints-1):
        camberLineSlope.append((yCamber[a+1] - yCamber[a]) / (xCamber[a+1] - xCamber[a]))
    camberLineSlope.append((yCamber[a+1] - yCamber[a]) / (xCamber[a+1] - xCamber[a]))

    tetha = []

    for a in range (0, numberOfPoints):
        tetha.append(np.arccos(1 - 2 * xCamber[a]))
    tetha.append(math.pi)

    for a in range(0, numberOfPoints):
        anot = anot + camberLineSlope[a] * (tetha[a+1] - tetha[a]) 

    anot = aoa - 1 / math.pi * anot

    for N in range(0, numberOfPoints):
        an.append(0)
        for n in range(0, numberOfPoints):
            an[N] = an[N] + camberLineSlope[n] * math.cos(N*tetha[n]) * (tetha[n+1] - tetha[n])
        an[N] = 2 / math.pi * an[N] 
    cl = math.pi * (2 * anot + an[1])
    cl2 = 0
    for a in range(0, numberOfPoints):
        cl2 = cl2 + camberLineSlope[a] * (math.cos(tetha[a]) - 1) * (tetha[a+1] - tetha[a])

    cl2 = 2 * math.pi * (aoa + cl2 / math.pi)

   
    jetCurve = []

    for a in range(0, numberOfPoints):
        jetCurve.append(yCamber[a])
    
    x = []
    jetCurveTetha = []
    Acoef = camberLineSlope[numberOfPoints-2] - aoa
    jetappendX, jetappendY, jetInteractionPoints = jetModel(camberLineSlope[-1], yCamber[-1], aoa)
    for a in range (0, jetInteractionPoints):
        jetCurve.append(jetappendY[a])
        if a > 0:
            jetCurveTetha.append((jetappendY[a] - jetappendY[a-1])/ (jetappendX[a] - jetappendX[a-1]))
        
    jetCurveTetha.append((jetappendY[-1] - jetappendY[-2])/ (jetappendX[-1] - jetappendX[-2]))

    x = jetappendX

    totalX = []
    totalX.extend(xCamber)
    totalX.extend(x)

    newAn = []
    tetha = []
    for a in range(0, numberOfPoints):
        tetha.append(a * math.pi / numberOfPoints)
    tetha.append(math.pi)

    deltaJ = rhoJet * velocityJet * velocityJet * heightJet - rho * fsv * fsv * heightJet
    deltaCj = -deltaJ / (rho * fsv * fsv * airfoilLen)
    if deltaCjOverride != 0:
        deltaCj = deltaCjOverride
    newAn.append(0)
    for a in range(0, numberOfPoints):
        for xPos in range(0, jetInteractionPoints -1):
            if(x[xPos] > (0.0000000001 + (airfoilLen / 2 * (1 - math.cos(tetha[a]))))) or (x[xPos] < (-0.000000001 + (airfoilLen / 2 * (1 - math.cos(tetha[a]))))):
                newAn[0] = newAn[0] +  1 / math.pi * (
                    deltaCj * airfoilLen / (4 * math.pi) * (jetCurveTetha[xPos+1] - jetCurveTetha[xPos])  / (x[xPos]/airfoilLen - 1 / 2 * (1 - math.cos(tetha[a]))) * (tetha[a+1] - tetha[a])
                    )
            else:
                newAn[0] = newAn[0] +  1 / math.pi * (
                    deltaCj * airfoilLen / (4 * math.pi) * (jetCurveTetha[xPos+1] - jetCurveTetha[xPos]) * (tetha[a+1] - tetha[a])
                    )   
        newAn[0] = newAn[0] - 1/math.pi * camberLineSlope[a] * (tetha[a+1] - tetha[a])
    newAn[0] = 1 * (aoa + newAn[0])

    for a in range(1, numberOfPoints):
        newAn.append(0)
        for b in range(0, numberOfPoints):
            for xPos in range(0, jetInteractionPoints-1):
                newAn[a] += (1 / math.pi * (
                    -1 * deltaCj  * airfoilLen / (2 * math.pi) * (jetCurveTetha[xPos+1] - jetCurveTetha[xPos]) / (0.000000001 +  x[xPos]/airfoilLen - 1 / 2 * (1 - math.cos(tetha[b]))  ) * (tetha[b+1] - tetha[b]) * math.cos(a * tetha[b]) 
                ))
            newAn[a] = newAn[a] + 2/math.pi * camberLineSlope[b] * (tetha[b+1] - tetha[b]) * math.cos(a * tetha[b])
    newAn = [number * 1 for number in newAn]
    clNew = math.pi * (2 * newAn[0] + newAn[1])
    return(clNew, cl)

#Conditions
fsv = 80 #free stream velocity in m/s
rho = 0.909
rhoJet = 0.816
percFlap = 0 #Percentage of the wing that is a flap
flapAngle = 0 * math.pi / 180 #angle of the flaps
propellerDia = 0.13
velocityJet = 95
airfoilLen = 0.1 #Length of the airfoil or the chord length
wingspan = 1.25
AoA = 0 * math.pi / 180  #Angle of attack in radians
#Flaps

#Jet
heightJet = (1/2) * (fsv / velocityJet + 1) * propellerDia #This is responsable for setting the height of the jet (this value should include the contraction effects of the wake)
deltaCjOverride = 0 #If this is non zero, the value of the calulated Cj will be overwritten

numberOfPoints = 0 # It is used to find the number of points in the CSV file which stores the chordline coordinates
normalizationVal = 0.01 #This is used to normalize all the x values of the airfoil such that they will be between 0 and 1
liftPerSpan = 0.0 #The output of the total lift that is generated per span is stored here
cl = 0.0 #The coefficient of lift

#Empties to get the camber angles
xCamber = []
yCamber = []

cl1, cl1old= twoDimentionalAirfoil(1*math.pi / 180)
cl2, cl2old= twoDimentionalAirfoil(2 * math.pi / 180)

slope = (cl2 - cl1) / (1 * math.pi / 180)
#we begin the 3D analysis beyond this part

def chordVarFunc(numberOfSpanControlPoints):
    span = []

    for a in range(0,numberOfSpanControlPoints):
        span.append(1)
    return(span)

def aoaVarFunc(numberOfSpanControlPoints):
    aoa = []

    for a in range(0, numberOfSpanControlPoints):
        aoa.append(AoA)
    return(aoa)

numberOfSpanControlPoints = 100
jetWidth = 5    #This is the width of the individual region that is under the effect of the jet
openWidth = 0    #This is the width of the individual region that is not under the effect of the jet
#Both above quantities are expressed in terms of wingspan / numberOfSpanPoints

Acoefs = []
tetha = []
chordVar = []
ar = wingspan / airfoilLen     #Aspect Ratio
chordVar = chordVarFunc(numberOfSpanControlPoints)
aoaKnot = []
slopeM = []
aoaVar = []
aoaVar = aoaVarFunc(numberOfSpanControlPoints)
b = 0
for a in range (0, numberOfSpanControlPoints):
    if b < openWidth:
        slopeM.append((cl2old - cl1old) / (1 * math.pi / 180))
        b+=1
    else:
        slopeM.append((cl2 - cl1) / (1 * math.pi / 180))
        b+=1
        if b == jetWidth + openWidth:
            b = 0
b = 0
for a in range(0, numberOfSpanControlPoints):
    if b < openWidth:
        aoaKnot.append(1 * math.pi / 180 - cl1old / slopeM[a])
        b+=1
    else:
        aoaKnot.append(1 * math.pi / 180 - cl1 / slopeM[a])
        b+=1
        if b == jetWidth + openWidth:
            b = 0

tetha = []
m0,alpha0=slopeM[1] , aoaKnot[1]
n=500   #Resilutiuon of th jet
rhs= aoaVar[1]-alpha0
rhs_mat=np.full((n,1),rhs)
for a in range(1, n+1):
    tetha.append(a * math.pi / n)
c_mat=[]
for beta in tetha:
    r_mat=[]
    for j in range(1,2*n,2):
        e=math.sin(j*beta)*((j/math.sin(beta))+(4*wingspan/(m0*airfoilLen)))
        r_mat.append(e)
    c_mat.append(r_mat)
coeff=np.linalg.solve(c_mat,rhs_mat)
cl=coeff[0]*wingspan*math.pi/airfoilLen
cdi=0
for i in range(0,len(coeff)):
    cdi+=(2*i+1)*(coeff[i]*coeff[i])*math.pi*wingspan/airfoilLen
print("Lift Coefficient CL:", cl)
print("Induced Drag Coefficient CDi:", cdi)
print("Lift to Drag Ratio L/D:", cl/cdi)
