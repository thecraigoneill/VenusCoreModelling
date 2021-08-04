from __future__ import absolute_import


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
import os
import sys  
sys.path.insert(0, '/Users/craig/Downloads/burnman-master/burnman/')

import burnman
from burnman import minerals
import burnman.mineral_helpers as helpers
from numba import jit 
import threading

# This file follows from the Burnman planet structure example. See burnman.org; documentation provided there.
# This code optimises the output MoI (and mass) to fit Venus observations.

# You only need this line if burnman is not already working at this point (otherwise ignore)
# !{sys.executable} -m pip install burnman

# From burman comments (obviously change the dir):
#      This is critical. We need 2 files here that have some routines we ABSOLUTELY NEED. 
#      These are planet.py and layer.py, which were included in the bundle. 
#      Whatever directory they are in - needs to be in the path next. It's good to have these files and your notebook in the one directory.
sys.path.insert(0, '/Users/craig/Downloads/burnman-master/burnman/')
#from burnman.layer import Layer
from burnman import layer
from burnman import planet

# Radius Venus 6051.8 km (c/w Earth 6371.km)
ratio =  6051.8/6371

radius_planet = 6371.e3*ratio
# Need to prefine layer positions for Burnman - the inner core and outer core radius will be varied each loop.
LM_rad=5711.e3*ratio
UM_rad=(6371.0 - 410.0)*1e3*ratio

#print("Initial radii guess:",IC_rad1/1e3,OC_rad1/1e3)
MoI_V = 0.337
Mass_V = 4.867e24/1e24
real = np.array([MoI_V, Mass_V])


# Returns the misfit MoI for Nelder Mead optimisation
def fit_MoI(x0):
    global sol_metal
    global liq_metal
    global rock
    global TZ_rock
    global UM_rock

    # inner_core
    IC_rad=abs(x0[0])
    OC_rad=abs(x0[1])
    if OC_rad > LM_rad:
        OC_rad = LM_rad - 1e3
    inner_core = layer.Layer('inner core', radii = np.linspace(0,IC_rad,10))
    inner_core.set_material(sol_metal)   
    # The minerals that make up our core do not currently implement the thermal equation of state, so we will set the temperature at 300 K.
    inner_core.set_temperature_mode('user-defined',300.*np.ones_like(inner_core.radii))

    # outer_core
    outer_core = layer.Layer('outer core', radii = np.linspace(IC_rad,OC_rad,10))
    outer_core.set_material(liq_metal)
    # The minerals that make up our core do not currently implement the thermal equation of state, so we will define the temperature at 300 K.
    outer_core.set_temperature_mode('user-defined', 300.*np.ones_like(outer_core.radii))

    # Next the Mantle.
    lower_mantle = layer.Layer('lower mantle', radii = np.linspace(OC_rad, LM_rad, 10))
    #lower_mantle.set_material(burnman.minerals.SLB_2011.mg_bridgmanite())
    lower_mantle.set_material(rock)
    lower_mantle.set_temperature_mode('adiabatic')

    # Transition zone
    transition_zone = layer.Layer('transition zone', radii = np.linspace(LM_rad, UM_rad, 10))
    transition_zone.set_material(TZ_rock)
    transition_zone.set_temperature_mode('adiabatic')


    # Upper mantle
    upper_mantle = layer.Layer('upper mantle', radii = np.linspace(UM_rad, radius_planet, 10))
    upper_mantle.set_material(UM_rock)
    upper_mantle.set_temperature_mode('adiabatic', temperature_top = T_adia)

    planet_zog = planet.Planet('Planet Zog',[inner_core, outer_core, lower_mantle, transition_zone, upper_mantle], verbose=False)
    planet_zog.make()
    diff1 = abs(MoI_V - planet_zog.moment_of_inertia_factor)
    calc = np.array([planet_zog.moment_of_inertia_factor,planet_zog.mass/1e24])
    #diff2 = np.linalg.norm(real - calc) 
    diff2 = np.sqrt( (real[0] - calc[0])**2 +  (real[1] - calc[1])**2)
    print('\t\t',IC_rad,OC_rad,diff2)
    return diff2

#Returns a MoI for a given structure
def return_MoI(x0):
    global sol_metal
    global liq_metal
    global rock
    global TZ_rock
    global UM_rock
     # inner_core
    IC_rad=abs(x0[0])
    OC_rad=abs(x0[1])
    inner_core = layer.Layer('inner core', radii = np.linspace(0,IC_rad,10))
    inner_core.set_material(sol_metal)   
    # The minerals that make up our core do not currently implement the thermal equation of state, so we will set the temperature at 300 K.
    inner_core.set_temperature_mode('user-defined',300.*np.ones_like(inner_core.radii))

    # outer_core
    outer_core = layer.Layer('outer core', radii = np.linspace(IC_rad,OC_rad,10))
    outer_core.set_material(liq_metal)
    # The minerals that make up our core do not currently implement the thermal equation of state, so we will define the temperature at 300 K.
    outer_core.set_temperature_mode('user-defined', 300.*np.ones_like(outer_core.radii))

    # Next the Mantle.
    lower_mantle = layer.Layer('lower mantle', radii = np.linspace(OC_rad, LM_rad, 10))
    #lower_mantle.set_material(burnman.minerals.SLB_2011.mg_bridgmanite())
    lower_mantle.set_material(rock)
    lower_mantle.set_temperature_mode('adiabatic')
    
    # Transition zone
    transition_zone = layer.Layer('transition zone', radii = np.linspace(LM_rad, UM_rad, 10))
    transition_zone.set_material(TZ_rock)
    transition_zone.set_temperature_mode('adiabatic')


    upper_mantle = layer.Layer('upper mantle', radii = np.linspace(UM_rad, radius_planet, 10))
    upper_mantle.set_material(UM_rock)
    upper_mantle.set_temperature_mode('adiabatic', temperature_top = T_adia)

    planet_zog = planet.Planet('Planet Zog',[inner_core, outer_core, lower_mantle, transition_zone, upper_mantle], verbose=False)
    planet_zog.make()
    #diff = abs(MoI_V - planet_zog.moment_of_inertia_factor)
    return(planet_zog)

#test = fit_MoI(1,IC_rad,OC_rad)
#print("Success",test)

IC_radii = np.array([])
OC_radii = np.array([])
MoI = np.array([])
VMass = np.array([])

n=100

bdg  = np.random.normal(0.8, 0.1, n)
Tadia = np.random.normal(1800,110,n)
IC_init = np.random.normal(700e3,300e3,n)
OC_init = np.random.normal(3000e3,500e3,n)
enst = np.random.uniform(0.13,0.4,n)
diop = np.random.uniform(0.1,0.17,n)
fost = 1 - enst - diop
wads = np.random.uniform(0.4,0.1,n)
ring = 1 - wads
FeSiCore = np.random.uniform(0.15,0.1,n)

# Examples for controlled experiments

#bdg = np.linspace(0.79,0.79,8) #06-0.8,n=8
#enst = np.random.uniform(0.2,0.2,8)
#diop = np.random.uniform(0.2,0.2,8)
#fost = 1 - enst - diop
#wads = np.linspace(0.4,0.4,8)
#FeSiCore = np.linspace(0.4,0.0,8)
#Tadia = np.linspace(1500,2100,10)
#Tadia = np.ones_like(bdg)*1800
#bdg = np.ones_like(Tadia)*0.9
#IC_init = np.ones_like(Tadia)*500e3
#OC_init = np.ones_like(Tadia)*3000e3

DSi=0.9 #Partitioning of Si into the solid core
CoreSolid_Si = DSi * FeSiCore


# Mod the following to test other variables
# @jit(nopython=True)
def loop_compositions():
  IC_radii = np.array([])
  OC_radii = np.array([])
  MoI = np.array([])
  VMass = np.array([])
  f=open('Venus_MC_results_w_SolidCore_Rev3.dat','a') # Opens an output file to append solutions to. 

  for i in range(len(bdg)):
    if(IC_init[i] < 0):
        IC_init[i] = 1.0
    if(OC_init[i] < IC_init[i]):
        IC_init[i] = OC_init[i] - 500.
    if (bdg[i] > 1.0):
        bdg[i] = 1.0
    if (FeSiCore[i] < 0.0):
        FeSiCore[i] = 0.0

    amount_perovskite = bdg[i]
    amount_fost = fost[i]
    amount_enst = enst[i]
    amount_diop = diop[i]
    amount_wads = wads[i]
    amount_ring = 1-amount_wads
    amount_FeSiCore = FeSiCore[i]
    amount_FeSiSolidCore = CoreSolid_Si[i]

    T_adia = Tadia[i]
    #print("In: Bdg%",bdg[i],"T_adia:",Tadia[i],"ICrad:",IC_init[i],"OCrad:",OC_init[i],LM_rad,radius_planet)
    print(i,'\t',amount_FeSiSolidCore,amount_FeSiCore)
    global sol_metal
    global liq_metal
    global rock
    global TZ_rock
    global UM_rock

    rock = burnman.Composite([minerals.SLB_2011.mg_perovskite(), minerals.SLB_2011.periclase()],[amount_perovskite, 1 - amount_perovskite])
    UM_rock = burnman.Composite( [burnman.minerals.SLB_2011.forsterite(), burnman.minerals.SLB_2011.enstatite(), burnman.minerals.SLB_2011.diopside()], [amount_fost, amount_enst, amount_diop])
    TZ_rock =  burnman.Composite([burnman.minerals.SLB_2011.mg_wadsleyite(),burnman.minerals.SLB_2011.mg_ringwoodite()],[amount_wads, amount_ring])
    liq_metal = burnman.Composite([burnman.minerals.other.Fe_Si(),burnman.minerals.other.Liquid_Fe_Anderson()],[amount_FeSiCore,1 - amount_FeSiCore])
    sol_metal = burnman.Composite([burnman.minerals.other.Solid_FeSi(),burnman.minerals.other.Fe_Dewaele()],[amount_FeSiSolidCore,1 - amount_FeSiSolidCore])

    IC = IC_init[i]
    OC = OC_init[i]
    # The following varies the inner core and outer core radius to fit the MoI, for the given Bridgmanite %, Adiabat, etc.
    result = minimize(fit_MoI,x0=[IC,OC],method = 'Nelder-Mead',tol=1e-6)
    planet_zog = return_MoI(result.x)
    IC_radii = np.append(IC_radii,abs(result.x[0]))
    OC_radii = np.append(OC_radii,abs(result.x[1]))
    MoI = np.append(MoI,planet_zog.moment_of_inertia_factor)
    VMass = np.append(VMass,planet_zog.mass)
    #print("Rad_IC,Rad_OC,MoI,VMass,grav:",result.x[0],result.x[1],planet_zog.moment_of_inertia_factor,planet_zog.mass,planet_zog.gravity[-1])
    #outstr = str(np.c_[abs(result.x[0]),abs(result.x[1]),planet_zog.moment_of_inertia_factor,planet_zog.mass,FeSiCore[i],bdg[i],wads[i],ring[i],fost[i],enst[i],diop[i],Tadia[i]])
    f=open('Venus_MC_results_w_SolidCore_Rev3.dat','a')
    np.savetxt(f,np.c_[abs(result.x[0]),abs(result.x[1]),planet_zog.moment_of_inertia_factor,planet_zog.mass,FeSiCore[i],bdg[i],wads[i],ring[i],fost[i],enst[i],diop[i],Tadia[i]])
    f.write('\n')
    f.close()
  return IC_radii, OC_radii, MoI, VMass 


IC_radii, OC_radii, MoI, VMass = loop_compositions()

# The following can save the output at the end of the loop

#np.savetxt("Venus_MC_results_w_SolidCore_Rev3.dat",np.c_[IC_radii,OC_radii,MoI,VMass,FeSiCore,bdg,wads,ring,fost,enst,diop,Tadia])

#print('\nmass/Earth= {0:.3f}, moment of inertia factor= {1:.4f}'.format(planet_zog.mass / 5.97e24, planet_zog.moment_of_inertia_factor))
#print(" MoI 0.337 +/- 0.024;  Venus mass 4.867 × 10^24 kg, 0.815M_Earth. \n")

