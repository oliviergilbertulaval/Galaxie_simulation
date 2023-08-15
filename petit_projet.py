from astropy.modeling.models import Gaussian2D
from photutils.datasets import make_noise_image
import numpy as np
import matplotlib.pyplot as plt
from random import *

def create_image(randomize=False):
    posx, posy = 400, 500
    angle = 40
    stdevx, stdevy = 20, 12
    if randomize:
        posx, posy = random()*500+300, random()*500+300
        angle = 360*random()
        stdevx, stdevy = 30*random(), 30*random()
    #Create the galaxy
    g = Gaussian2D(100.0, posx, posy, stdevx, stdevy, theta=angle * np.pi / 180.0)
    ny = nx = 1000
    y, x = np.mgrid[0:ny, 0:nx]
    noise = make_noise_image((ny, nx), distribution='gaussian', mean=0.0, stddev=1.0, seed=None)
    data = g(x, y) + noise

    return data

def show_image(img):
    plt.imshow(img, cmap='gray')
    plt.show()

galaxie = create_image(randomize=False)
show_image(galaxie)