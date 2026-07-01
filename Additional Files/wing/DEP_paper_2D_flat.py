'''
Here we try to implement the whatever we have learnt in the thin airfoil theory to the DEP case
'''

import math

velJet = 50
heightJet = 0.15 * (1/2) * (20 / velJet + 1) #This is responsable for setting the height of the jet (this value should include the contraction effects of the wake)
rhoJet = 1.225
rhoinf = 1.225
chord = 1
Vinf = 20
aoa = 7.5 * math.pi / 180
jetDeflection = 30 * math.pi / 180

jDash = rhoJet * velJet * velJet * heightJet
cj = jDash / (rhoinf * Vinf * Vinf * chord)
cl = 2 * math.pi * (1 + 0.151 * math.sqrt(cj) + 0.219 * cj) * aoa + 2 * math.sqrt(math.pi * cj) * math.sqrt(1 + 0.151 * math.sqrt(cj) + 0.139 * cj) * jetDeflection

print(cl)