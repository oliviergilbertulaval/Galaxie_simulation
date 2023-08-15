from astropy.modeling.models import Gaussian2D
from photutils.datasets import make_noise_image
import numpy as np
import matplotlib.pyplot as plt
from random import *
from projet_classe import MockImage

#On créé l'image avec la classe
image = MockImage()

#Utilise image.show() pour voir l'image avec matplotlib
#Ex:
image.show()

#Utilise image.data pour accéder au tableau numpy de tes données
#Ex:
print(image.data)

