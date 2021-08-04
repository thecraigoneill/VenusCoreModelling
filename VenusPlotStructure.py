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
import sys  

# You only need this line if burnman is not already working at this point (otherwise ignore)
# !{sys.executable} -m pip install burnman

# This is critical. We need 2 files here that have some routines we ABSOLUTELY NEED. 
#These are planet.py and layer.py, which were included in the bundle. 
# Whatever directory they are in - needs to be in the path next. It's good to have these files and your notebook in the one directory.
sys.path.insert(0, '/Users/craig/Downloads/burnman-master/burnman/')
#from burnman.layer import Layer
from burnman import layer
from burnman import planet

# Radius Venus 6051.8 km (c/w Earth 6371.km)
ratio =  6051.8/6371

radius_planet = 6371.e3*ratio
IC_rad1=1220.e3*ratio
OC_rad1=3480.e3*ratio

LM_P = 3300*9.81*660e3
LM_depth_Venus = LM_P/(3300*8.87)
print("UM depth:",LM_depth_Venus)

#LM_rad=5711.e3*ratio
LM_rad = radius_planet - LM_depth_Venus 

LM_rad=5711.e3*ratio
UM_rad=(6371.0 - 410.0)*1e3*ratio


T_adia=1800

#print("Initial radii guess:",IC_rad1/1e3,OC_rad1/1e3)
MoI_V = 0.337
Mass_V = 4.867e24/1e24
real = np.array([MoI_V, Mass_V])

#amount_perovskite=0.9
#rock = burnman.Composite([minerals.SLB_2011.mg_perovskite(), minerals.SLB_2011.periclase()],[amount_perovskite, 1 - amount_perovskite])


# Returns the misfit MoI for Nelder Mead optimisation
def fit_MoI(x0):
    # inner_core
    IC_rad=abs(x0[0])
    OC_rad=abs(x0[1])
    if OC_rad > LM_rad:
        OC_rad = LM_rad - 1e3
    inner_core = layer.Layer('inner core', radii = np.linspace(0,IC_rad,10))
    inner_core.set_material(burnman.minerals.other.Fe_Dewaele())   
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
    print(IC_rad,OC_rad,diff2)
    return diff2

#Returns a MoI for a given structure
def return_MoI(x0):
     # inner_core
    IC_rad=abs(x0[0])
    OC_rad=abs(x0[1])
    inner_core = layer.Layer('inner core', radii = np.linspace(0,IC_rad,10))
    inner_core.set_material(burnman.minerals.other.Fe_Dewaele())   
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
bnds = ((0., 3000), (0, 4000.0e3))
#x0 = np.array([3,3])
#result = minimize(fit_MoI,x0=[500e3,3000e3],method = 'Nelder-Mead',tol=1e-7)
#IC_rad1,OC_rad1
#J = result.jac
#print(J,"Jacobian")
#cov = np.linalg.inv(J.T.dot(J))
#var = np.sqrt(np.diagonal(cov))
## print(result)
#print("Variance",var)
#print("Inner Core/Out core radius:", result.x/1e3)
#x00=np.array([-5.57925358e-01,  3.12851423e+06])
#x00 = np.array([1, 3.12851423e+06])
#x00 = np.array([500e3, 3.17160066e+06])
#x00=result.x
#print("MoI,  Ma[]ss:",return_MoI(result.x))
#print(x00[1],LM_rad)
#metal = burnman.minerals.other.Liquid_Fe_Anderson()

#planet_zog = return_MoI(x00)

#moi2=planet_zog.moment_of_inertia_factor
#mass2=planet_zog.mass
#print("MoI,  Mass:",moi2,mass2)

amount_FeSiCore = 0.0
amount_perovskite=0.8
amount_fost=0.6
amount_enst=0.2
amount_diop=0.2
amount_wads=0.5
amount_ring=0.5
x00 = np.array([9.475842080565747283e-01, 3.120057171202933416e+06])

rock = burnman.Composite([minerals.SLB_2011.mg_perovskite(), minerals.SLB_2011.periclase()],[amount_perovskite, 1 - amount_perovskite])
#metal = burnman.minerals.other.Liquid_Fe_Anderson()
UM_rock = burnman.Composite( [burnman.minerals.SLB_2011.forsterite(), burnman.minerals.SLB_2011.enstatite(), burnman.minerals.SLB_2011.diopside()], [amount_fost, amount_enst, amount_diop])
TZ_rock =  burnman.Composite([burnman.minerals.SLB_2011.mg_wadsleyite(),burnman.minerals.SLB_2011.mg_ringwoodite()],[amount_wads, amount_ring])
liq_metal = burnman.Composite([burnman.minerals.other.Fe_Si(),burnman.minerals.other.Liquid_Fe_Anderson()],[amount_FeSiCore,1 - amount_FeSiCore])

planet_Fe = return_MoI(x00)
print('\nmass/Earth= {0:.3f}, moment of inertia factor= {1:.4f}'.format(planet_Fe.mass / 5.97e24, planet_Fe.moment_of_inertia_factor))
print('\n Gravity=',(planet_Fe.gravity[-1]),'vs 8.87',(planet_Fe.gravity[-1])/8.87)

amount_FeSiCore = 0.4
amount_perovskite=0.8
amount_fost=0.6
amount_enst=0.2
amount_diop=0.2
x01 = np.array([2.220449380554244854e+06, 4.482874292045780458e+06])

rock = burnman.Composite([minerals.SLB_2011.mg_perovskite(), minerals.SLB_2011.periclase()],[amount_perovskite, 1 - amount_perovskite])
#metal = burnman.minerals.other.Liquid_Fe_Anderson()
UM_rock = burnman.Composite( [burnman.minerals.SLB_2011.forsterite(), burnman.minerals.SLB_2011.enstatite(), burnman.minerals.SLB_2011.diopside()], [amount_fost, amount_enst, amount_diop])
TZ_rock =  burnman.Composite([burnman.minerals.SLB_2011.mg_wadsleyite(),burnman.minerals.SLB_2011.mg_ringwoodite()],[amount_wads, amount_ring])
liq_metal = burnman.Composite([burnman.minerals.other.Fe_Si(),burnman.minerals.other.Liquid_Fe_Anderson()],[amount_FeSiCore,1 - amount_FeSiCore])
planet_FeSi =  return_MoI(x01)

print('\nmass/Earth= {0:.3f}, moment of inertia factor= {1:.4f}'.format(planet_FeSi.mass / 5.97e24, planet_FeSi.moment_of_inertia_factor))
print(" MoI 0.337 +/- 0.024;  Venus mass 4.867 × 10^24 kg, 0.815M_Earth. \n")
#print('\n Gravity={0:0.3f}'.format(planet_FeSi.gravity))
print('\n Gravity=',(planet_FeSi.gravity[-1]),'vs 8.87',(planet_FeSi.gravity[-1])/8.87)



#print(planet_zog.temperature)
 
figure = plt.figure(figsize=(17, 9))
plt.rcParams.update({'font.size': 18})

ax = [figure.add_subplot(1, 5, i) for i in range(1, 6)]

#ax[0].plot(planet_zog.density / 1.e3, planet_zog.radii / 1.e3, color="xkcd:forest", linewidth=2.)
ax[0].plot(planet_Fe.density / 1.e3, planet_Fe.radii / 1.e3, color="xkcd:midnight blue", linewidth=2.,label="Fe core")
ax[0].plot(planet_FeSi.density / 1.e3, planet_FeSi.radii / 1.e3, color="xkcd:burnt red", linewidth=2.,label="FeSi core")
ax[0].legend(fontsize=11,loc="upper right")

print("density difference",np.c_[planet_FeSi.density,((planet_Fe.density - planet_FeSi.density)/planet_Fe.density)] )


#ax[0].plot( premradii / 1.e3, premdensity / 1.e3, '--k', linewidth=1.,
#        label='PREM')
ax[0].set_xlim(0., (max(planet_Fe.density) / 1.e3) + 1.)
ax[0].set_xlabel('Density ($\cdot 10^3$ kg/m$^3$)')
ax[0].set_ylabel('Radius (km)')
ax[0].set_ylim(0,6051.8)

#print(np.c_[planet_zog.radii / 1.e3,planet_zog.density / 1.e3, planet_Fe.density,planet_FeSi.density])

#print(np.c_[planet_Fe.radii / 1.e3, planet_Fe.density,planet_FeSi.density])

# Make a subplot showing the calculated pressure profile
ax[1].plot(planet_Fe.pressure / 1.e9, planet_Fe.radii / 1.e3,   color="xkcd:midnight blue", linewidth=2.)
ax[1].plot(planet_FeSi.pressure / 1.e9, planet_FeSi.radii / 1.e3,   color="xkcd:burnt red", linewidth=2.)

#ax[1].plot(premradii / 1.e3, prempressure / 1.e9, '--b', linewidth=1.)
ax[1].set_xlim(0., (max(planet_Fe.pressure) / 1e9) + 10.)
ax[1].set_xlabel('Pressure (GPa)')
ax[1].set_ylim(0,6051.8)

# Make a subplot showing the calculated gravity profile
ax[2].plot(planet_Fe.gravity, planet_Fe.radii / 1.e3,  color="xkcd:midnight blue", linewidth=2.)
ax[2].plot(planet_FeSi.gravity, planet_FeSi.radii / 1.e3,  color="xkcd:burnt red", linewidth=2.)

#ax[2].plot(premradii / 1.e3, premgravity, '--g', linewidth=1.)
ax[2].set_xlabel('Gravity (m/s$^2)$')
ax[2].set_xlim(0., max(planet_Fe.gravity) + 0.5)
ax[2].set_ylim(0,6051.8)

# Make a subplot showing the calculated temperature profile
P= planet_Fe.pressure[planet_Fe.radii <= 3.120057171202933416e+06]/1e9

planet_Fe.temperature[ planet_Fe.radii <= 3.120057171202933416e+06] = 2292.5 + 12.645*P - 7.9689e-3*P**2


ax[3].plot( planet_Fe.temperature, planet_Fe.radii / 1.e3,  color="xkcd:midnight blue", linewidth=2.,label="Fe Core: T")

#Melting Fe from Steinbrugge 2020
#Tm = 495.5e9* ((22.2e9 - planet_FeSi.pressure)**0.42)
# From Anzellini 2013

Ptp = 95.5
Ttp=3712

P2= planet_Fe.pressure/1e9
Tm2 = ( ((P2 - Ptp)/161.2 + 1)**(1/1.72)) * Ttp
filt2 =  planet_Fe.radii <= x00[1]
ax[3].plot(Tm2[filt2], planet_Fe.radii[filt2] / 1.e3,  color="xkcd:midnight blue", linewidth=2.,linestyle="--",label="Melting T Fe")
Xes=0.11 + 0.187*np.exp(-0.065*P2)
Tes = 1346 - 13.0*(P2 - 21.0)
Xs = 0.0106
Tm_FeS = Tm2 - ((Tm2-Tes)/Xes)*Xs
ax[3].plot(Tm_FeS[filt2], planet_Fe.radii[filt2] / 1.e3,  color="xkcd:midnight blue", linewidth=2.,linestyle="-.",label="Melting T FeS")

ax[3].fill_betweenx(planet_Fe.radii[filt2]/1e3,Tm2[filt2],Tm_FeS[filt2],  color="xkcd:midnight blue",alpha=0.25)
ax[3].legend(fontsize=10,loc="upper right")
ax[3].set_xlabel('Temperature ($K$)')
#ax[3].set_ylabel('Pressure (GPa)')
ax[3].set_ylim(0,6051.8)


P= planet_FeSi.pressure[ planet_Fe.radii <= 3.120057171202933416e+06]/1e9
planet_FeSi.temperature[ planet_FeSi.radii <= 4.482874292045780458e+06] = 2292.5 + 12.645*P - 7.9689e-3*P**2
ax[4].plot( planet_FeSi.temperature, planet_FeSi.radii / 1.e3,  color="xkcd:burnt red", linewidth=2.,label="FeSi Core: T")

P= planet_FeSi.pressure/1e9
Tm = ( ((P - Ptp)/161.2 + 1)**(1/1.72)) * Ttp
#Tm=2292.5 + 12.645*P - 7.9689e-3*P**2
filt = planet_FeSi.radii <= x01[1]
ax[4].plot(Tm[filt], planet_FeSi.radii[filt] / 1.e3,  color="xkcd:burnt red", linewidth=2.,linestyle="--",label="Melting T Fe")


# FeS solidus - from Steinbrugge again. Xs~0.0106 is for Earth
#Possible Venus has more (if really enriched in light elements)
Xes=0.11 + 0.187*np.exp(-0.065*P)
Tes = 1346 - 13.0*(P - 21.0)
Xs = 0.0106
Tm_FeS = Tm - ((Tm-Tes)/Xes)*Xs
#print("Solidus FeS",np.c_[Tm_FeS,planet_FeSi.pressure,Tm,planet_Fe.pressure])

ax[4].plot(Tm_FeS[filt], planet_FeSi.radii[filt] / 1.e3,  color="xkcd:burnt red", linewidth=2.,linestyle="-.",label="Melting T FeS")
filt3 = ( planet_FeSi.radii <= x01[1])&(planet_FeSi.radii >= x01[0])
#ax[4].fill_betweenx(planet_FeSi.radii[filt3]/1e3,Tm_FeS[filt3],planet_FeSi.temperature[filt3],  color="xkcd:burnt red",alpha=0.25)

filt4 = (planet_FeSi.radii <= x01[1])
ax[4].fill_betweenx(planet_FeSi.radii[filt4]/1e3,Tm[filt4],Tm_FeS[filt4],  color="xkcd:burnt red",alpha=0.25)

ax[4].legend(fontsize=10,loc="upper right")
ax[4].set_xlabel('Temperature ($K$)')
#ax[4].set_ylabel('Pressure (GPa)')

#ax[3].set_xlim(1200., max(planet_Fe.temperature) + 100)
ax[4].set_ylim(0, 6051.8)

#for i in range(3):
#    ax[i].set_xticklabels([])
#for i in range(4):
#ax[i].set_xlim(0., max(planet_zog.radii) / 1.e3)
plt.tight_layout()    
plt.savefig("Venus_optimised_MoI_FeSi_Rev3a.png")
plt.show()
