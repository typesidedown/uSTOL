import numpy as np
import math
from extraction import getvalue

c=float(input("Enter Chord Length:"))
b=float(input("Enter Span Length:"))
Q=float(input("Enter Free Stream Velocity:"))
AoA=float(input("Enter Angle of Attack:"))

cl,aoa=getvalue()
m0,alpha_l0=np.polyfit(cl,aoa,1)
m0,alpha0 = (1/m0),alpha_l0

alpha0=-3.031
n=1000
rhs= math.radians(AoA-alpha0)
rhs_mat=np.full((n,1),rhs)
theta=np.linspace(1,180,n)
c_mat=[]
for beta in theta:
    r_mat=[]
    for j in range(1,2*n,2):
        e=math.sin(j*math.radians(beta))*((j/math.sin(math.radians(beta)))+(4*b/(m0*c)))
        r_mat.append(e)
    c_mat.append(r_mat)
#print(c_mat)
coeff=np.linalg.solve(c_mat,rhs_mat)
cl=coeff[0]*b*math.pi/c
cdi=0
for i in range(0,len(coeff)):
    cdi+=(2*i+1)*(coeff[i]*coeff[i])*math.pi*b/c
print(cl,"and", cdi)
