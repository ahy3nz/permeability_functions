import pdb
import glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import permeability_functions.thermo_functions as thermo_functions
import permeability_functions.misc as misc
import numpy as np
import simtk.unit as u
n_windows = 35
reaction_coordinates = np.loadtxt('z_windows.txt') * u.nanometer
window_forces = []
window_facf_integrals = []
for i, _ in enumerate(reaction_coordinates):
    print('window {}'.format(i))
    #data = np.loadtxt('forceout{}.txt'.format(i))
    #times = data[:,1] * u.femtosecond
    #forces = data[:,2] * u.kilocalorie/(u.mole*u.angstrom)
    #mean_force , time_intervals, facf = thermo_functions.analyze_force_timeseries(
    #        times, forces, 
    #        'meanforce{}.txt'.format(i), 'fcorr{}.txt'.format(i))
    mean_force = np.loadtxt('meanforce{}.txt'.format(i)) * u.kilocalorie/(u.mole*u.angstrom)
    times_facf = np.loadtxt('fcorr{}.txt'.format(i))
    time_intervals = times_facf[:,0] * u.picosecond
    facf = times_facf[:,1] * mean_force.unit **2

    intF, intFval = thermo_functions.integrate_facf_over_time(time_intervals, facf)
    window_forces.append(mean_force)
    window_facf_integrals.append(intFval)

window_forces = misc.validate_quantity_type(window_forces, mean_force.unit)

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

# Plotting
import pdb; pdb.set_trace() 
fig, ax = plt.subplots(2,1)
sym_fe_profile, _ = misc.symmetrize(fe_profile._value, zero_boundary_condition=True)
sym_fe_profile *= fe_profile.unit
ax[0].plot(reaction_coordinates._value, fe_profile._value, label='non-sym')
ax[0].set_xlabel("Reaction Coordinate ({})".format(reaction_coordinates.unit))
ax[0].set_ylabel("Free Energy ({})".format(fe_profile.unit))
ax[0].plot(reaction_coordinates._value, sym_fe_profile._value, label='sym')
ax[0].legend()


diffusion_profile= diffusion_profile.in_units_of(u.centimeter**2/u.second)
ax[1].semilogy(reaction_coordinates._value, diffusion_profile._value)
ax[1].set_xlabel("Reaction Coordinate ({})".format(reaction_coordinates.unit))
ax[1].set_ylabel("Diffusion ({})".format(diffusion_profile.unit))
fig.tight_layout()
fig.savefig('profiles.png')

