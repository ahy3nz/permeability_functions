import pdb
import glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import permeability_functions.thermo_functions as thermo_functions
import permeability_functions.misc as misc
import numpy as np
import pandas as pd
import simtk.unit as u
import os
def main():
    curr_dir = os.getcwd()

    #all_sweeps = [thing for thing in os.listdir() if os.path.isdir(thing) and 'sweep2' not in thing and 'sweep8' not in thing and 'sweep6' not in thing and '__pycache__' not in thing]
    all_sweeps = [thing for thing in os.listdir() if os.path.isdir(thing) and '__pycache__' not in thing]
    n_sims = 6
    reaction_coordinates = np.loadtxt('z_windows.out') * u.nanometer
    n_windows = len(reaction_coordinates)
    window_forces = np.zeros(n_windows) 
    window_facf_integrals = np.zeros(n_windows)
    df = pd.DataFrame()
    for sweep in all_sweeps:
        os.chdir(os.path.join(curr_dir, sweep))
        (reaction_coordinate, fe_profile) = (np.loadtxt('free_energy_profile.dat')[:,0],
                                            np.loadtxt('free_energy_profile.dat')[:,1])

        diffusion_profile = np.loadtxt('diffusion_profile.dat')[:,1]
        reaction_coordinate *=  u.nanometer
        fe_profile *= u.kilocalorie/ (u.mole)
        diffusion_profile *= (u.centimeter**2)/u.second
        res_profile, res_integral = thermo_functions.compute_resistance_profile(
                                            fe_profile, 
                                            diffusion_profile,
                                            reaction_coordinates)
        permeability_integral = thermo_functions.compute_permeability(res_integral)
        permeability_integral = permeability_integral.in_units_of(u.centimeter/u.second)
        print(sweep, permeability_integral)
        to_add = {'sweep': [sweep], 'permeability': [permeability_integral._value],
                'permeability_unit': [permeability_integral.unit]}

        temp_df = pd.DataFrame.from_dict(to_add)
        df = df.append(temp_df)

    os.chdir(curr_dir)
    df.to_csv("permeability_summary.csv")




if __name__ == "__main__":
    main()
