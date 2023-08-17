from typing import Any
from astropy.modeling.models import Gaussian2D
from photutils.datasets import make_noise_image
import numpy as np
import matplotlib.pyplot as plt
from random import *
from scipy import ndimage
from skimage.restoration import richardson_lucy

class PSF():
    def __init__(self, telescope='HST'):
        if telescope == 'HST':
            self.data = np.loadtxt(r'hubble_PSF.txt', dtype=int)
        else:
            raise ValueError("There is no PSF available for this telescope.")



class MockImage():
    def __init__(self, gal={'pos':(400,500), 'angle':40, 'stdev':(55,32), 'amplitude':30, 'agn':0}, agn_amp=None, psf = None, randomize=False, number_of_galaxies=1, add_noise=True, noise_level=5, noise_deviation=1.0):
        '''
        gal: dictionnary; to personnalize the first galaxy created (randomize has to be False)
        psf: PSF object; point spread function used to convolve the components
        randomize: boolean; wether or not to randomize the first galaxy
        number_of_galaxies: int; number of galaxies generated in the mock image
        add_noise: boolean; wether or not to add noise after convolving
        noise_level: float; mean of the noise
        noise_deviation: float; standard deviation of the noise
        agn_amp: float; amplitude of the first galaxy's AGN
        
        '''
        self.pos, self.angle, self.stdev, self.amplitude, self.agn, g, point_amp = [], [], [], [], [], [], []
        for i in range(number_of_galaxies):
            self.pos.append((random()*900+50, random()*900+50))
            self.angle.append(360*random())
            self.amplitude.append(100*random()+50)
            self.stdev.append((20*random()+5, 20*random()+5))
            self.agn.append(1 if random()>0.5 else 0)
            point_amp.append(random()*50+20)
            
            if randomize is False and i == 0:
                self.pos[0] = (gal['pos'])
                self.angle[0] = (gal['angle'])
                self.stdev[0] = (gal['stdev'])
                self.amplitude[0] = (gal['amplitude'])
                self.agn[0] = (gal['agn'])
                if agn_amp is not None:
                    point_amp[0] = agn_amp
            g.append(Gaussian2D(self.amplitude[i], self.pos[i][0], self.pos[i][1], self.stdev[i][0], self.stdev[i][1], theta=self.angle[i] * np.pi / 180.0))
            ran_seed = random()
            
            self.agn[i] = Gaussian2D(point_amp[i]*self.agn[i], self.pos[i][0], self.pos[i][1], 2, 2, theta=0)
        ny = nx = 1000
        y, x = np.mgrid[0:ny, 0:nx]
        noise = make_noise_image((ny, nx), distribution='gaussian', mean=noise_level, stddev=noise_deviation, seed=None)
        self.data = np.zeros(noise.shape)
        for i in range(number_of_galaxies):
            #On ajoute les galaxies à l'image
            self.data += g[i](x,y) + self.agn[i](x,y)
        
        #Convolve with PSF:
        if psf is not None:
            self.data = ndimage.convolve(self.data, psf.data, mode='reflect')/3134.28857895659

        #On ajoute le bruit dans l'image
        if add_noise:
            self.data += noise
        pass


    def show(self):
        #Affiche l'image à l'écran
        plt.imshow(self.data, cmap='gray', origin='lower')
        plt.show()
        pass

if __name__ == '__main__':

    #On créé un PSF
    psf = PSF('HST')

    #On créé une fake image avec 10 galaxies comme on la recevrait d'un télescope
    image_avec_10_galaxies = MockImage(randomize=True, number_of_galaxies=10, gal={'pos':(400,500), 'angle':40, 'stdev':(1,1), 'amplitude':100}, add_noise=True, noise_level=50, noise_deviation=2, psf=psf)
    image_avec_10_galaxies.show()