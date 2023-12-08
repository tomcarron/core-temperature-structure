import astropy.constants as const
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import time
import matplotlib.pylab as plb
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from spectral_cube import SpectralCube
from spectral_cube import Projection
import radio_beam
from radio_beam import Beam
from astropy.io import fits
from radmc3dPy.image import *    
from radmc3dPy.analyze import *  
from radmc3dPy.natconst import * 
from astropy.wcs import WCS
import sys

class model_setup_1Dspher:
    def __init__(self,rho0,prho,nphot=100000,rin=1000,rout=5000,gaussian=False,plateau=false):
        self.rho0=rho0
        self.nphot=nphot
        self.gaussian=gaussian
        # Grid parameters
        #
        au  = 1.49598e13# AU [cm]
        pc  = 3.08572e18     # Parsec                  [cm]
        ms  = 1.98892e33     # Solar mass              [g]
        ts  = 5.78e3         # Solar temperature       [K]
        ls  = 3.8525e33      # Solar luminosity        [erg/s]
        self.ls=ls
        rs  = 6.96e10        # Solar radius            [cm]
        self.nx       = 400
        self.ny       = 1
        self.nz       = 1
        self.sizex    = 2*rout*au
        self.sizey    = 2*rout*au
        self.sizez    = 2*rout*au
        
        # Model parameters
        #
        self.rin      = rin*au
        self.rout     = rout*au
        self.r_break = 50*au #edge of constant density region
        #rho0     = 1e8
        self.prho     = prho        #-2.e0
        #
        # Star parameters
        #
        self.mstar    = 30*ms
        self.rstar    = 13.4*rs
        self.tstar    = 5*ts
        self.pstar    = np.array([0.,0.,0.]) #position in cartesian coords

        #inner edge refinement parameters
        self.nlev_r=8
        self.nspan_r=10

                #
        # Make the coordinates
        #
        self.xi       = self.rin * (self.rout/self.rin)**(np.linspace(0.,self.nx,self.nx+1)/(self.nx-1.0))
        #self.xi     = self.rin*(np.logspace(np.log10(rin),np.log10(rout),self.nx+1))
        #self.xi       = np.linspace(self.rin,self.rout,self.nx+1)  
        #self.xi       = self.grid_refine_inner_edge(xi,self.nlev_r,self.nspan_r)   # Refinement at inner edge
        self.yi       = np.array([0.,math.pi])
        self.zi       = np.array([0.,math.pi*2])
        self.xc       = 0.5e0 * ( self.xi[0:self.nx] + self.xi[1:self.nx+1] )
        #
        # Make the dust density model
        #
        self.rr = self.xc
        self.rhod = rho0 * ((self.rr) / au) ** prho
        if plateau:
            self.rr = self.xc
            self.rhod = np.zeros_like(self.xc)
            for i in range(len(self.rhod)):
                if self.xc[i] < self.r_break+1*au: #hardcoded fix, need better
                    self.rhod[i] = rho0
                else:
                    self.rhod[i] = rho0 * ((self.rr[i] - self.r_break) / au) ** prho

        elif gaussian:
            self.rhod = rho0 * gaussian_profile(self.rr/au,1e-5)

        #
        # Write the wavelength_micron.inp file
        #
        lam1     = 0.1e0
        lam2     = 7.0e0
        lam3     = 25.e0
        lam4     = 1.0e4
        lam5	 = 9.0e4
        n12      = 100
        n23      = 100
        n34      = 100
        n45	 = 100
        lam12    = np.logspace(np.log10(lam1),np.log10(lam2),n12,endpoint=False)
        lam23    = np.logspace(np.log10(lam2),np.log10(lam3),n23,endpoint=False)
        lam34    = np.logspace(np.log10(lam3),np.log10(lam4),n34,endpoint=False)
        lam45    = np.logspace(np.log10(lam4),np.log10(lam5),n45,endpoint=True)
        self.lam      = np.concatenate([lam12,lam23,lam34,lam45])
        self.nlam     = self.lam.size


    def write_input(self,amr=False,mrw=False):
        # Write the wavelength file
    
        with open('wavelength_micron.inp','w+') as f:
            f.write('%d\n'%(self.nlam))
            for value in self.lam:
                f.write('%13.6e\n'%(value))
        #
        #
        # Write the stars.inp file
        #
        with open('stars.inp','w+') as f:
            f.write('2\n')
            f.write('1 %d\n\n'%(self.nlam))
            f.write('%13.6e %13.6e %13.6e %13.6e %13.6e\n\n'%(self.rstar,self.mstar,self.pstar[0],self.pstar[1],self.pstar[2]))
            for value in self.lam:
                f.write('%13.6e\n'%(value))
            f.write('\n%13.6e\n'%(-self.tstar))
        #
        # Write the grid file
        #
        if amr==False:
            with open('amr_grid.inp','w+') as f:
                f.write('1\n')                       # iformat
                f.write('0\n')                       # AMR grid style  (0=regular grid, no AMR)
                f.write('100\n')                     # Coordinate system spherical
                f.write('0\n')                       # gridinfo
                f.write('1 0 0\n')                   # Include x,y,z coordinate 1D spherically symmetric
                f.write('%d %d %d\n'%(self.nx,self.ny,self.nz))     # Size of grid
                for value in self.xi:
                    f.write('%13.6e\n'%(value))      # X coordinates (cell walls)
                for value in self.yi:
                    f.write('%13.6e\n'%(value))      # Y coordinates (cell walls)
                for value in self.zi:
                    f.write('%13.6e\n'%(value))      # Z coordinates (cell walls)
        else:
            self.generate_amr_grid_input(self.rout,50,50,5)


        #
        # Write the density file
        #
        with open('dust_density.inp','w+') as f:
            f.write('1\n')                       # Format number
            f.write('%d\n'%(self.nx*self.ny*self.nz))           # Nr of cells
            f.write('1\n')                       # Nr of dust species
            data = self.rhod.ravel(order='F')         # Create a 1-D view, fortran-style indexing
            data.tofile(f, sep='\n', format="%13.6e")
            f.write('\n')
        #
        # Dust opacity control file
        #
        with open('dustopac.inp','w+') as f:
            f.write('2               Format number of this file\n')
            f.write('1               Nr of dust species\n')
            f.write('============================================================================\n')
            f.write('1               Way in which this dust species is read\n')
            f.write('0               0=Thermal grain\n')
            f.write('silicate        Extension of name of dustkappa_***.inp file\n')
            f.write('----------------------------------------------------------------------------\n')
        #
        # Write the radmc3d.inp control file
        #
        with open('radmc3d.inp','w+') as f:
            f.write('nphot = %d\n'%(self.nphot))
            f.write('scattering_mode_max = 0\n')   # Put this to 1 for isotropic scattering
            f.write('iranfreqmode = 1\n')
            if mrw:
                f.write('modified_random_walk = 1')

    def calculate_model(self):
        t0 = time.time()
        os.system('radmc3d mctherm')
        #os.system('radmc3d sed')
        t1 = time.time()

        total = t1-t0
        print("Calculating the model cost: "+str(total)+" s")
        with open('cost.txt') as f:
            f.write("Calculating the model cost: "+str(total)+" s")
        #Make the necessary calls to run radmc3d
        return
        
    def sed(self):
        #plot sed
        s    = readSpectrum()
        plt.figure()
        lammic = s[:,0]
        flux   = s[:,1]
        nu     = 1e4*const.c.cgs/lammic
        nufnu  = nu*flux
        nulnu  = nufnu*4*math.pi*(const.pc.cgs)*(const.pc.cgs)
        plt.plot(lammic,nulnu/self.ls)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\lambda$ [$\mu$m]')
        plt.ylabel(r'$\nu L_{\nu}$ [$L_{\odot}$]')
        plt.axis()
        plt.show()
    
    def make_synth_maps(self,wls):
        t0 = time.time()
        for wl in wls:
            os.system('radmc3d image lambda '+str(wl))
            im   = readImage()
            im.writeFits('model'+str(wl)+'.fits')
            process_radmc_image('model'+str(wl)+'.fits','model'+str(wl)+'_smooth.fits',0.4)
            cim = im.imConv(fwhm=[0.4, 0.4], pa=0., dpc=8340.) 
            #plt.figure()
            plotImage(cim, arcsec=True, dpc=8340., cmap=plb.cm.gist_heat)
            #cim.writeFits('model'+str(wl)+'.fits', dpc=8340.)
            #plt.show()
        t1 = time.time()
        total=t1-t0
        print("Calculating the images cost: "+str(total)+" s")
        #plot synthetic maps at each wavelength specified.

    def make_tau_surface(self,wls):
        for wl in wls:
            os.system("radmc3d tausurf 1.0 lambda "+str(wl))
            im   = readImage()
            #plotImage(im,arcsec=True,dpc=8340.)
            im.writeFits('tau_surf_'+str(wl)+'.fits')

    def make_tau_map(self,wls):
        for wl in wls:
            os.system("radmc3d image lambda "+str(wl)+" tracetau")
            im   = readImage()
            #plotImage(im,arcsec=True,dpc=8340.)
            im.writeFits('tau'+str(wl)+'.fits')

    def make_column_density_map(self,wls):
        for wl in wls:
            os.system("radmc3d image lambda "+str(wl)+" tracecolumn")
            im   = readImage()
            #plotImage(im,arcsec=True,dpc=8340.)
            im.writeFits('column_density_'+str(wl)+'.fits')

    
    def temperature_profile(self):
        #plot temperature vs radius
        au=const.au.cgs.value # AU [cm]
        a    = readData(dtemp=True,binary=False)
        r    = a.grid.x[:]
        temp = a.dusttemp[:,0,0,0]
        plt.figure(1)
        plt.plot(r/au,temp)
        plt.xlabel('r [au]')
        plt.ylabel('T [K]')
        plt.show()
        plt.savefig('temp.png')

    def masses(self):
        a    = readData(binary=False)
        dust_mass=a.getDustMass()
        #gas_mass=a.getGasMass(mweight=2.3,rhogas=True)
        print("Dust mass :", dust_mass /1.989e33)
        #print("Gas mass :", gas_mass /1.989e33)
        #print('Total mass :', (dust_mass+gas_mass)/1,989e33)

    
    def density_profile(self):
        #plot density vs radius
        au=const.au.cgs.value # AU [cm]
        a    = readData(ddens=True,binary=False)
        r    = a.grid.x[:]
        density = a.rhodust[:,0,0,0]
        plt.figure(1)
        plt.plot(r/au,density)
        plt.xlabel('r [au]')
        plt.ylabel(r'$\rho_{dust}$ [$g/cm^3$]')
        plt.show()
        plt.savefig('density.png')

    def tau(self):
        #THIS NEEDS TO BE PLOTTED FOR A GIVEN WAVELENGTH
        #optical depth
        au=const.au.cgs.value # AU [cm]
        a    = readData(binary=False)
        r    = a.grid.x[:]
        density = a.taux#[:,0,0,0]
        plt.figure(1)
        plt.plot(r/au,density)
        plt.xlabel('r [au]')
        plt.ylabel(r'rho_d [g/cm^3]')
        plt.show()

    def grid_refine_inner_edge(self,x_orig,nlev,nspan):
        x     = x_orig.copy()
        rev   = x[0]>x[1]
        for ilev in range(nlev):
            x_new = 0.5 * ( x[1:nspan+1] + x[:nspan] )
            x_ref = np.hstack((x,x_new))
            x_ref.sort()
            x     = x_ref
            if rev:
                x = x[::-1]
        return x
    
    def generate_amr_grid_input(self,radius_max, nr_outer, nr_refine, amr_level_max):
        """
        Generates amr_grid.inp file for a 1D spherically symmetric model with adaptive mesh refinement (AMR).

        Parameters:
        - radius_max: Maximum radius of the model (in AU).
        - nr_outer: Number of radial grid points in the outer region (coarse grid).
        - nr_refine: Number of additional radial grid points in each refinement level.
        - amr_level_max: Maximum AMR refinement level.

        Outputs:
        - Writes the amr_grid.inp file.
        """

        with open("amr_grid.inp", "w") as f:
            # Write header
            #f.write("1D\n")
            #f.write("sph\n")
            f.write("1\n\n")  # Coordinate system: spherical
            f.write("1\n")      #refinement
            #f.write('0\n')                       # gridinfo
            #f.write('1 0 0\n')                   # Include x,y,z coordinate 1D spherically symmetric

            # Write grid parameters
            f.write(f"{nr_outer}\n")  # Number of radial grid points in the outer region
            f.write(f"{radius_max}\n")  # Maximum radius of the model (in AU)
            f.write(f"{amr_level_max}\n\n")  # Maximum AMR refinement level

            # Write refinement level parameters
            for i in range(1, amr_level_max + 1):
                f.write(f"{i}\n")  # Refinement level
                f.write(f"{nr_refine}\n")  # Number of additional radial grid points in each refinement level
                f.write("1\n\n")  # Number of cells in the azimuthal and polar direction

            # Write closing statement
            f.write("-1\n")

def smooth_fits_image(input_file, output_file, target_resolution_major, target_resolution_minor):
    '''
    smooths an image to a target resolution.
    '''
    # Open the input FITS file
    hdul = fits.open(input_file)
    data = hdul[0].data

    # Get the current beam resolution
    current_resolution_major = hdul[0].header['BMAJ']  # Assumes major axis beam resolution is stored in BMAJ keyword
    current_resolution_minor = hdul[0].header['BMIN']  # Assumes minor axis beam resolution is stored in BMIN keyword

    # Compute the kernel width ratios
    sigma_ratio_major = target_resolution_major / current_resolution_major
    sigma_ratio_minor = target_resolution_minor / current_resolution_minor

    # Create a Gaussian kernel for smoothing
    kernel = Gaussian2DKernel([sigma_ratio_major, sigma_ratio_minor])

    # Convolve the data with the kernel
    smoothed_data = convolve(data, kernel)

    # Update the header with the new beam resolution
    hdul[0].header['BMAJ'] = target_resolution_major
    hdul[0].header['BMIN'] = target_resolution_minor

    # Save the smoothed data to a new FITS file
    hdul[0].data = smoothed_data
    hdul.writeto(output_file, overwrite=True)

    # Close the input FITS file
    hdul.close()


def process_radmc_image(input_fits, output_fits,beam_size_arcsec,overwrite=False):
    hdulist = fits.open(input_fits)
    data = extract_dimensions(hdulist[0].data)
    header = hdulist[0].header
    wcs = WCS(header)

    # Extract pixel size information from the WCS. Pixel sizes are given in degrees
    pixel_size_x, pixel_size_y = wcs.pixel_scale_matrix[1, 1], wcs.pixel_scale_matrix[0, 0]

    # Convert beam size from arcseconds to pixels
    beam_size_x = abs(int(beam_size_arcsec / 3600 / pixel_size_x))
    beam_size_y = abs(int(beam_size_arcsec / 3600 / pixel_size_y))

    #check if beam sizes are odd *must be odd for kernel
    if beam_size_x % 2 == 0:
        beam_size_x+=1
    if beam_size_y % 2 == 0:
        beam_size_y+=1

    # Calculate the standard deviation of the Gaussian kernel
    beam_stddev_x = beam_size_x / (2 * np.sqrt(2 * np.log(2)))

    print(beam_size_x,beam_size_y)
    # Create a 2D Gaussian kernel
    kernel = Gaussian2DKernel(beam_stddev_x, x_size=int(beam_size_x), y_size=int(beam_size_y))

    # Convolve the data with the Gaussian kernel
    smoothed_data = convolve(data, kernel, normalize_kernel=True)

    # Scale the entire image to convert pixel values from Jy/pixel to Jy/beam
    #conversion_factor = ((pixel_size_x/3600) * (pixel_size_y/3600) ) / (np.pi * beam_size_arcsec**2)
    #smoothed_data *= conversion_factor

    # Update the header to reflect the new units and beam information
    header['BUNIT'] = 'Jy/beam'
    header['BMAJ'] = beam_size_arcsec
    header['BMIN'] = beam_size_arcsec

    # Save the smoothed data to a new FITS file
    fits.writeto(output_fits, smoothed_data, header, overwrite=overwrite)

    print(f"Smoothing completed. Result saved to: {output_fits}")

def gaussian_profile(r,alpha):
    return np.exp(-alpha * r**2)

#function which returns the two largest dimensions of an array
def extract_dimensions(array):
    if array.ndim <= 2:
        return array
    else:
        dimensions_to_remove = np.where(np.array(array.shape) < 2)[0]
        modified_array = np.squeeze(array, axis=tuple(dimensions_to_remove))
        return modified_array
