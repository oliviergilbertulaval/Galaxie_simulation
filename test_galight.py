import numpy as np
import matplotlib.pyplot as plt
from projet_classe import *
import astropy.io.fits as pyfits
import galight.tools.astro_tools as astro_tools
from galight.data_process import DataProcess, cut_center_auto
from galight.fitting_specify import FittingSpecify
from galight.fitting_process import FittingProcess

psf = PSF('HST')
if False:
    image = MockImage(
        randomize=False, 
        number_of_galaxies=1, 
        star=True, 
        gal={'pos':(400,500), 'angle':40, 'stdev':(5,10), 'amplitude':0.003, 'agn':1}, 
        agn_amp=0.005, 
        psf=psf, 
        add_noise=True, 
        noise_level=0.0007, 
        noise_deviation=0.003
        )
    image = MockImage(
        randomize=False, 
        number_of_galaxies=1, 
        star=True, 
        gal={'pos':(400,500), 'angle':70, 'stdev':(10,12), 'amplitude':0.001, 'agn':1}, 
        agn_amp=0.1, 
        psf=psf, 
        add_noise=True, 
        noise_level=0.0007, 
        noise_deviation=0.003
        )
image = MockImage(
    randomize=True, 
    number_of_galaxies=1, 
    star=True,
    psf=psf, 
    if_agn='all',
    add_noise=True, 
    noise_level=0.0007, 
    noise_deviation=0.003
    )
image.show()
#fitsFile = pyfits.open(image)




#Load the fov image data:
fov_image = image.data # check the back ground


#Derive the header informaion, might be used to obtain the pixel scale and the exposure time.
#header = fitsFile_qso[1].header # if target position is add in WCS, the header should have the wcs information, i.e. header['EXPTIME']

#Load the WHT map, which would be use to derive the exposure map
#wht = fitsFile_qso[2].data


#exp =  astro_tools.read_fits_exp(fitsFile_qso[0].header)  #Read the exposure time 
exp = 2260.0
wht = np.ones(image.data.shape)*exp
mean_wht = exp * (0.0642/0.135)**2  #The drizzle information is used to derive the mean WHT value.
exp_map = exp * wht/mean_wht  #Derive the exposure time map for each pixel





#keywords see the notes above.
data_process = DataProcess(fov_image = fov_image, target_pos = image.pos[0], pos_type = 'pixel', header = None,
                          rm_bkglight = False, exptime = exp_map, if_plot=False, zp = 24.897)  #zp use 27.0 for convinence.

data_process.generate_target_materials(radius=50, create_mask = False, nsigma=5,
                                      exp_sz= 1.2, npixels = 15, if_plot=True)

data_process.find_PSF(radius = 30, PSF_pos_list=[[800, 800]], user_option = True)  #Try this line out! 
#data_process.find_PSF(radius = 30, user_option = True)

#Plot the FOV image and label the position of the target and the PSF
data_process.plot_overview(label = 'Example', target_label = None)

# Compare the 1D profile of all the components.
data_process.profiles_compare(norm_pix = 5, if_annuli=False, y_log = False,
                  prf_name_list = (['target'] + ['PSF{0}'.format(i) for i in range(len(data_process.PSF_list))]) )

#Select which PSF id you want to use to make the fitting.
#data_process.psf_id_for_fitting = int(input('Use which PSF? Input a number.\n'))
data_process.psf_id_for_fitting = 0

#Check if all the materials is given, if so to pass to the next step.
data_process.checkout()




#PREPARE THE FITTING

data_process.deltaPix = 0.03962


fit_sepc = FittingSpecify(data_process)

#Prepare the fitting sequence, keywords see notes above.
fit_sepc.prepare_fitting_seq(point_source_num = 1, fix_n_list= None, fix_center_list = [[0, 0]], 
                            extend_source_model=None, source_params = None, ps_params = None)

#Plot the initial settings for fittings. 
fit_sepc.plot_fitting_sets()

#Build up and to pass to the next step.
fit_sepc.build_fitting_seq()




#Setting the fitting method and run.


#Pass fit_sepc to FittingProcess,
# savename: The name of the saved files.    
fit_run = FittingProcess(fit_sepc, savename = 'HST_mock', fitting_level='deep') 



#For the fitting_level, you can also put ['norm', 'deep'] for the later ['PSO', 'MCMC'] corresplingly.

#Setting the fitting approach and Run: 
#     algorithm_list: The fitting approaches that would be used: e.g. ['PSO', 'PSO', 'MCMC']
#     setting_list: The detailed setting for the fitting:
#     -for PSO:
#         input template: {'sigma_scale': 0.8, 'n_particles': 50, 'n_iterations': 50}
#     -for MCMC:
#         input template: {'n_burn': 50, 'n_run': 100, 'walkerRatio': 10, 'sigma_scale': .1}
#     if setting_list = [None, None, None], default values would be given 
fit_run.run(algorithm_list = ['PSO', 'MCMC'], setting_list=None)
#fit_run.run(algorithm_list = ['PSO', 'MCMC'], setting_list = [None, {'n_burn': 200, 'n_run': 1000, 'walkerRatio': 10, 'sigma_scale': .1}])

# Plot all the fitting results, including:
#         run_diag() : The convergence of the chains.
#         model_plot(): The model plot (by lenstronomy)
#fit_run.plot_params_corner() #: The mcmc corner for all the chains (MCMC should be peformed) 
#         plot_flux_corner(): The flux corner for all the component (MCMC should be peformed)
#         plot_final_qso_fit() or plot_final_galaxy_fit(): Plot the overall plot (data, model, data-ps, resudal, 1D profile)
fit_run.plot_all(target_ID = 'Mock_Image')

#Save the fitting class as pickle format:
#     Note, if you use python3 (or 2), load with python3 (or 2)
fit_run.dump_result()