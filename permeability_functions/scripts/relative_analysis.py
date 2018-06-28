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

def main():
    n_sims = 5
    local_tuples = []
    leaflet_tuples = []

    for sim_number in range(n_sims):
        traj = mdtraj.load('Sim{0}/trajectory.dcd'.format(sim_number), 
                            top='Sim{0}/Stage4_Eq{0}.gro'.format(sim_number))
        tracers = np.loadtxt('Sim{0}/tracers.out'.format(sim_number), dtype=int) - 1

        d_from_local_i_list, d_from_leaflet_i_list = grid_funcs.distance_from_interface(
                                                                    traj, tracers)
        d_from_local_i_list = misc.validate_quantity_type(d_from_local_i_list, 
                                                            u.nanometer)
        d_from_leaflet_i_list = misc.validate_quantity_type(d_from_leaflet_i_list, 
                                                            u.nanometer)
        # Need to relate distance from interface to tracer to forceout index
        for i, (tracer, d_from_local_i, d_from_leaflet_i ) in enumerate(zip(
                                                tracers,
                                                d_from_local_i_list, d_from_leaflet_i_list)):
            forceout_id = sim_number + (i*n_sims)
            data = np.loadtxt('Sim{0}/forceout{1}.dat'.format(sim_number, forceout_id))
            times = data[:,1] * u.femtosecond
            forces = data[:,2] * u.kilocalorie/(u.mole*u.angstrom)
            mean_force , time_intervals, facf = thermo_functions.analyze_force_timeseries(
                    times, forces, 
                    meanf_name='Sim{0}/meanforce{1}.dat'.format(sim_number, i), 
                    fcorr_name='Sim{0}/fcorr{1}.dat'.format(sim_number, i))
            intF, intFval = thermo_functions.integrate_facf_over_time(time_intervals, facf)
            local_tuples.append((d_from_local_i, mean_force, time_intervals, intFval))
            leaflet_tuples.append((d_from_leaflet_i, mean_force, time_intervals, intFval))

    local_tuples = (sorted(local_tuples, key=lambda stuff: stuff[0]), 
                        'local_interface')
    leaflet_tuples = (sorted(leaflet_tuples, key=lambda stuff: stuff[0]), 
                        'leaflet_interface')

    for (tuple_thing, suffix) in [local_tuples, leaflet_tuples]:
        reaction_coordinates, mean_forces, time_intervals, facf_integrals = zip(
                                                                            *tuple_thing)

        (reaction_coordinates, mean_forces, facf_integrals, fe_profile, 
                diffusion_profile, resistance_profile, resistance_integral, 
                permeability_profile, permeability_integral) = thermo_functions.permeability_routine(reaction_coordinates, mean_forces, facf_integrals)

        np.savetxt('diffusion_profile_{}.dat'.format(suffix), 
                    np.column_stack((reaction_coordinates, diffusion_profile)))
        np.savetxt('free_energy_profile_{}.dat'.format(suffix), 
                    np.column_stack((reaction_coordinates, fe_profile)))
        np.savetxt('resistance_profile_{}.dat'.format(suffix), 
                    np.column_stack((reaction_coordinates, resistance_profile)))
        np.savetxt('permeability_profile_{}.dat'.format(suffix), 
                    np.column_stack((reaction_coordinates, permeability_profile)))

        print(permeability_integral)

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
        fig.savefig('profiles_{}.png'.format(suffix))
        plt.close(fig)
if __name__ == "__main__":
    main()
