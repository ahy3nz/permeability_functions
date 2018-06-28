import pdb
import glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import mdtraj
import numpy as np
import simtk.unit as u

import permeability_functions.thermo_functions as thermo_functions
import permeability_functions.grid_functions as grid_funcs
import permeability_functions.misc as misc
n_sims = 5
sim_number =0 
traj = mdtraj.load('Sim0/trajectory.dcd', top='Sim0/Stage4_Eq0.gro')
tracers = np.loadtxt('Sim0/tracers.out', dtype=int) - 1
n_tracers = len(tracers)

d_from_local_i_list, d_from_leaflet_i_list = grid_funcs.distance_from_interface(
                                                            traj, tracers)
d_from_local_i_list = misc.validate_quantity_type(d_from_local_i_list, 
                                                    u.nanometer)
d_from_leaflet_i_list = misc.validate_quantity_type(d_from_leaflet_i_list, 
                                                    u.nanometer)
# Need to relate distance from interface to tracer to forceout index
window_tuples = []
for i, (tracer, d_from_local_i, d_from_leaflet_i ) in enumerate(zip(
                                        tracers,
                                        d_from_local_i_list, d_from_leaflet_i_list)):
    forceout_id = sim_number + (i*n_sims)
    data = np.loadtxt('forceout{}.txt'.format(forceout_id))
    times = data[:,1] * u.femtosecond
    forces = data[:,2] * u.kilocalorie/(u.mole*u.angstrom)
    mean_force , time_intervals, facf = thermo_functions.analyze_force_timeseries(
            times, forces, 
            meanf_name='meanforce{}.txt'.format(i), 
            fcorr_name='fcorr{}.txt'.format(i))
    intF, intFval = thermo_functions.integrate_facf_over_time(time_intervals, facf)
    window_tuples.append((d_from_local_i, mean_force, time_intervals, intFval))

window_tuples = sorted(window_tuples, key=lambda stuff: stuff[0])
reaction_coordinates, mean_forces, time_intervals, facf_integrals = zip(
                                                                    *window_tuples)
reaction_coordinates = misc.validate_quantity_type(reaction_coordinates, 
                                                reaction_coordinates[0].unit)
mean_forces = misc.validate_quantity_type(mean_forces, mean_force.unit)

facf_integrals = misc.validate_quantity_type(facf_integrals, 
                                                intFval.unit)

fe_profile = thermo_functions.compute_free_energy_profile(mean_forces,
                                                        reaction_coordinates)

diffusion_profile = thermo_functions.compute_diffusion_coefficient(
                                                        facf_integrals)

resistance_profile, resistance_integral = thermo_functions.compute_resistance_profile(
                                                fe_profile, 
                                                diffusion_profile, 
                                                reaction_coordinates)
permeability_profile = thermo_functions.compute_permeability(resistance_profile)
permeability_profile = permeability_profile.in_units_of(u.centimeter**2/u.second)
permeability_integral = thermo_functions.compute_permeability(resistance_integral)
permeability_integral = permeability_integral.in_units_of(u.centimeter/u.second)

# Plotting
fig, ax = plt.subplots(2,1)
ax[0].plot(reaction_coordinates._value, fe_profile._value, label='non-sym')
ax[0].set_xlabel("Reaction Coordinate ({})".format(reaction_coordinates.unit))
ax[0].set_ylabel("Free Energy ({})".format(fe_profile.unit))
ax[0].legend()


diffusion_profile= diffusion_profile.in_units_of(u.centimeter**2/u.second)
ax[1].semilogy(reaction_coordinates._value, diffusion_profile._value)
ax[1].set_xlabel("Reaction Coordinate ({})".format(reaction_coordinates.unit))
ax[1].set_ylabel("Diffusion ({})".format(diffusion_profile.unit))
fig.tight_layout()
fig.savefig('profiles_local_interface.png')
