import pdb
import glob
import permeability_functions.thermo_functions as thermo_functions
import numpy as np
import simtk.unit as u
n_windows = 35
data = np.loadtxt('forceout0.txt')
times = data[:,1] * u.femtosecond
forces = data[:,2] * u.kilocalorie/(u.mole*u.angstrom)
mean_force , time_intervals, facf = thermo_functions.analyze_force_timeseries(times, forces, 'meanforce.txt', 'fcorr.txt')
intF, intFval = thermo_functions.integrate_facf_over_time(time_intervals, facf)
window_forces = [mean_force] * n_windows
reaction_coordinates = np.linspace(0, n_windows*2 + 1, num=n_windows) * u.angstrom

fe_profile = thermo_functions.compute_free_energy_profile(window_forces,
                                                reaction_coordinates)
int_F_vals =[intFval] * n_windows
diffusion_profile = thermo_functions.compute_diffusion_coefficient(int_F_vals)

resistance_profile = thermo_functions.compute_resistance_profile(fe_profile, 
                                                diffusion_profile, 
                                                reaction_coordinates)
permeability_profile = thermo_functions.compute_permeability_profile(resistance_profile)
pdb.set_trace()
