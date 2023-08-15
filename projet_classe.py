from typing import Any
from astropy.modeling.models import Gaussian2D
from photutils.datasets import make_noise_image
import numpy as np
import matplotlib.pyplot as plt
from random import *

class MockImage():
    def __init__(self, gal={'pos':(400,500), 'angle':40, 'stdev':(20,12), 'amplitude':100}, randomize=False, number_of_galaxies=1, noise_level=5):
        
        self.pos, self.angle, self.stdev, self.amplitude, g = [], [], [], [], []
        for i in range(number_of_galaxies):
            self.pos.append((random()*900+50, random()*900+50))
            self.angle.append(360*random())
            self.amplitude.append(100*random()+50)
            self.stdev.append((20*random()+5, 20*random()+5))
            if randomize is False and i == 0:
                self.pos[0] = (gal['pos'])
                self.angle[0] = (gal['angle'])
                self.stdev[0] = (gal['stdev'])
                self.amplitude[0] = (gal['amplitude'])
            g.append(Gaussian2D(self.amplitude[i], self.pos[i][0], self.pos[i][1], self.stdev[i][0], self.stdev[i][1], theta=self.angle[i] * np.pi / 180.0))

        #Create the galaxy(ies)
        ny = nx = 1000
        y, x = np.mgrid[0:ny, 0:nx]
        noise = make_noise_image((ny, nx), distribution='gaussian', mean=noise_level, stddev=1.0, seed=None)
        self.data = noise
        for i in range(number_of_galaxies):
            self.data += g[i](x,y)
        pass

    def show(self):
        plt.imshow(self.data, cmap='gray')
        plt.show()
        pass

image = MockImage(randomize=True, number_of_galaxies=10)
image.show()