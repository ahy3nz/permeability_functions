import os
import pdb
import glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import permeability_functions.thermo_functions as thermo_functions
import permeability_functions.misc as misc
import numpy as np
import simtk.unit as u
def main():
    all_sweeps = [thing for thing in os.listdir() is os.path.isdir(thing) and 'cache' not in thing]
    curr_dir = os.getcwd()
    n_sims = 6
    for sweep in all_sweeps:
        print(sweep)
        os.chdir(os.path.join(curr_dir, sweep))
        reaction_coordinates = np.loadtxt('z_windows.out') * u.nanometer
        n_windows = len(reaction_coordinates)
        window_forces = np.zeros(n_windows)
        window_facf_integrals = np.zeros(n_windows)
        for sim_number in range(n_sims):
            tracers = np.loadtxt('Sim{0}/tracers.out'.format(sim_number), dtype=int) - 1
            for i,tracerid in enumerate(tracers):
                forceout_id = sim_number + (i*n_sims)
                data = np.loadtxt('Sim{0}/condensed_forceout{1}.dat'.format(sim_number, forceout_id))
                times = data[:,0] * u.femtosecond
                forces = data[:,1] * u.kilocalorie/(u.mole*u.angstrom)
                mean_force , time_intervals, facf = thermo_functions.analyze_force_timeseries(
                        times, forces, 
                        meanf_name='Sim{0}/meanforce{1}.dat'.format(sim_number, forceout_id), 
                        fcorr_name='Sim{0}/fcorr{1}.dat'.format(sim_number, forceout_id))
                intF, intFval = thermo_functions.integrate_facf_over_time(time_intervals, 
                                                                            facf)
                window_forces[forceout_id] = mean_force._value
                window_facf_integrals[forceout_id] = intFval._value
        
        
        (reaction_coordinates, mean_forces, facf_integrals, fe_profile, 
                    diffusion_profile, resistance_profile, resistance_integral, 
                    permeability_profile, permeability_integral) = thermo_functions.permeability_routine(reaction_coordinates, window_forces, window_facf_integrals)
        
        np.savetxt('diffusion_profile.dat', np.column_stack((reaction_coordinates, diffusion_profile)))
        np.savetxt('free_energy_profile.dat', np.column_stack((reaction_coordinates, fe_profile)))
        np.savetxt('resistance_profile.dat', np.column_stack((reaction_coordinates, resistance_profile)))
        np.savetxt('permeability_profile.dat', np.column_stack((reaction_coordinates, permeability_profile)))
        
        
        print(permeability_integral)
    
    # Plotting
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
    plt.close(fig)
    
if __name__ == "__main__":
    main()
