import pdb
import glob
import permeability_functions.thermo_functions as thermo_functions
import permeability_functions.misc as misc
import numpy as np
import simtk.unit as u
n_windows = 35
reaction_coordinates = np.loadtxt('z_windows.txt') * u.angstrom
window_forces = []
window_facf_integrals = []
for i, _ in enumerate(reaction_coordinates):
    print('window {}'.format(i))
    data = np.loadtxt('forceout{}.txt'.format(i))
    times = data[:,1] * u.femtosecond
    forces = data[:,2] * u.kilocalorie/(u.mole*u.angstrom)
    mean_force , time_intervals, facf = thermo_functions.analyze_force_timeseries(
            times, forces, 
            'meanforce{}.txt'.format(i), 'fcorr{}.txt'.format(i))
    intF, intFval = thermo_functions.integrate_facf_over_time(time_intervals, facf)
    window_forces.append(mean_force)
    window_facf_integrals.append(intFval)
window_forces = misc.validate_quantity_type(window_forces, forces.unit)

window_facf_integrals = misc.validate_quantity_type(window_facf_integrals, 
                                                intFval.unit)

fe_profile = thermo_functions.compute_free_energy_profile(window_forces,
                                                        reaction_coordinates)

diffusion_profile = thermo_functions.compute_diffusion_coefficient(
                                                        window_facf_integrals)

resistance_profile = thermo_functions.compute_resistance_profile(fe_profile, 
                                                diffusion_profile, 
                                                reaction_coordinates)
permeability_profile = thermo_functions.compute_permeability_profile(resistance_profile)
pdb.set_trace()
