import sys
sys.path.append('../')
from temperature_structure.model_setup import model_setup_1Dspher as model
import astropy.constants as const


rho0=1e10 *1.6735575e-24   #reference density cgs units
prho=-1.0 #power law index

#def gaussian_profile(r,alpha):
#    return np.exp(-alpha * r**2)

#print(gaussian_profile())

#initialise model
test=model(rho0,prho,nphot=100000,rin=10,rout=2000,gaussian=False,plateau=False)

#write input file
test.write_input(amr=False,mrw=True)

#run model
test.calculate_model()

#plot sed
#test.sed()

#plot temperature profile
test.temperature_profile()

#plot density profile
test.density_profile() #dont work

#test.masses()

#opacities
#test.tau()

#create synthetic maps 
wls=[450,850,1000,2000,3000] #micron
test.make_synth_maps(wls)
#test.make_tau_surface(wls)


