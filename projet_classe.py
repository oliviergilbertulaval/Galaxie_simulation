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
        self.telescope = telescope
        if telescope == 'HST':
            self.data = np.loadtxt(r'hst_PSF.txt', dtype=float)
            #self.show()
        else:
            raise ValueError("There is no PSF available for this telescope.")
        
    def show(self):
        plt.imshow(self.data)
        plt.suptitle(self.telescope)
        plt.show()



class MockImage():
    def __init__(self, gal={'pos':(400,500), 'angle':40, 'stdev':(55,32), 'amplitude':30, 'agn':0}, star=False, if_agn='maybe', agn_amp=None, psf = None, randomize=False, number_of_galaxies=1, add_noise=True, noise_level=5, noise_deviation=1.0):
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
            self.amplitude.append(0.001*random()+0.0002)
            self.stdev.append((15*random()+5, 15*random()+5))
            if if_agn == 'maybe':
                self.agn.append(1 if random()>0.5 else 0)
            elif if_agn == 'all':
                self.agn.append(1)
            else:
                self.agn.append(0)
            point_amp.append(random()*5+0.015)
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
            
            self.agn[i] = Gaussian2D(point_amp[i]*self.agn[i], self.pos[i][0], self.pos[i][1], 0.1, 0.1, theta=0)
        ny = nx = 1000
        y, x = np.mgrid[0:ny, 0:nx]
        noise = make_noise_image((ny, nx), distribution='gaussian', mean=noise_level, stddev=noise_deviation, seed=None)
        self.data = np.zeros(noise.shape)
        for i in range(number_of_galaxies):
            #On ajoute les galaxies à l'image
            self.data += g[i](x,y) + self.agn[i](x,y)
        if star:
            self.data += Gaussian2D(1, 800, 800, 0.1, 0.1, theta=0)(x,y)
        #Convolve with PSF:
        if psf is not None:
            self.data = ndimage.convolve(self.data, psf.data, mode='reflect')

        #On ajoute le bruit dans l'image
        if add_noise:
            self.data += noise
        pass


    def show(self):
        #Affiche l'image à l'écran
        ax1 = plt.subplot(111)
        ax1.imshow(self.data, cmap='gray', origin='lower', vmax=0.01)
        ticklabels = ax1.get_xticklabels()
        ticklabels.extend(ax1.get_yticklabels())
        for label in ticklabels:
            label.set_fontsize(14)
        ax1.set_ylabel(r'pixel', size=16)
        ax1.set_xlabel(r'pixel', size=16)
        #plt.suptitle('SDSS J101152.98+544206.4, z = 0.246', size=16)
        plt.show()
        pass

if __name__ == '__main__':

    #On créé un PSF
    psf = PSF('HST')
    #psf.show()

    #On créé une fake image avec 10 galaxies comme on la recevrait d'un télescope
    image_avec_10_galaxies = MockImage(randomize=True, number_of_galaxies=10, gal={'pos':(400,500), 'angle':40, 'stdev':(1,1), 'amplitude':100}, add_noise=True, noise_level=50, noise_deviation=2, psf=psf)
    image_avec_10_galaxies.show()