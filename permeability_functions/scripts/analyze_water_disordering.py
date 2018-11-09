import os
import glob
import subprocess
import pdb
import mdtraj
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import plot_ay
plot_ay.setDefaults()
import bilayer_analysis_functions

###############################
## From permeability simulations,
## compute bilayer properties over time
###############################

def main():
    curr_dir = os.getcwd()
    all_nums = np.arange(0,20, dtype=int)
    all_simnums = np.arange(0,5, dtype=int)
    all_sweeps = ['sweep{}'.format(num) for num in all_nums]
    all_sims = ['Sim{}'.format(num) for num in all_simnums]

    all_s2 = []
    all_apt = []
    for sweep in all_sweeps:
        for sim in all_sims:
            os.chdir(os.path.join(curr_dir, sweep, sim))
            print("Converting in {}".format((sweep,sim)))
            trajfile, grofile = prepare_traj(trajname='combined_nopbc.xtc')
            traj = mdtraj.load(trajfile, top=grofile)
            print("Analyzing in {}".format((sweep,sim)))
            s2list = compute_disorder(traj)
            aptlist = compute_packing(traj)
            all_s2.append(s2list)
            all_apt.append(aptlist)

    os.chdir(curr_dir)

    all_s2 = prune_arrays(all_s2)
    fig, ax = plt.subplots(1,1)
    frames = np.arange(all_s2.shape[1])
    l, = ax.plot(frames, np.mean(all_s2, axis=0))
    ax.fill_between(frames, 
            np.mean(all_s2, axis=0) - np.std(all_s2, axis=0),
            np.mean(all_s2, axis=0) + np.std(all_s2, axis=0),
            color=l.get_color(), alpha=0.4)
    ax.set_xlabel("Frame")
    ax.set_ylabel("S2")
    plot_ay.tidyUp(fig, ax, gridArgs={}, tightLayoutArgs={})
    fig.savefig('permeation_order.png')
    np.savetxt("s2_permeation.dat", np.column_stack((frames, 
        np.mean(all_s2, axis=0),
        np.std(all_s2,axis=0))))
    plt.close(fig)

    all_apt = prune_arrays(all_apt)
    fig, ax = plt.subplots(1,1)
    frames = np.arange(all_apt.shape[1])
    l, = ax.plot(frames, np.mean(all_apt, axis=0))
    ax.fill_between(frames, 
            np.mean(all_apt, axis=0) - np.std(all_apt, axis=0),
            np.mean(all_apt, axis=0) + np.std(all_apt, axis=0),
            color=l.get_color(), alpha=0.4)
    ax.set_xlabel("Frame")
    ax.set_ylabel("APT [$\AA^2$]")
    plot_ay.tidyUp(fig, ax, gridArgs={}, tightLayoutArgs={})
    fig.savefig('permeation_apt.png')
    np.savetxt("apt_permeation.dat", np.column_stack((frames, 
        np.mean(all_apt, axis=0),
        np.std(all_apt,axis=0))))
    plt.close(fig)

def compute_disorder(traj):
    """ Measure S2 frame by frame"""
    if not os.path.isfile('s2_permeation.dat'):
        _, _, s2list = bilayer_analysis_functions.calc_nematic_order(traj, 
                blocked=False)
        np.savetxt('s2_permeation.dat', s2list)
    else:
        s2list = np.loadtxt('s2_permeation.dat')
    return s2list

def compute_packing(traj):
    """ Measure APT frame by frame """
    if not os.path.isfile("apt_permeation.dat"):
        lipid_tails, _ = bilayer_analysis_functions.identify_groups(traj,
                forcefield='charmm36')
        n_lipid = len([res for res in traj.topology.residues if not res.is_water])
        n_lipid_tails = len(lipid_tails.keys())
        n_tails_per_lipid = n_lipid_tails/n_lipid

        _,_, apl_list = bilayer_analysis_functions.calc_APL(traj, n_lipid, 
                blocked=False)
        _,_, angle_list = bilayer_analysis_functions.calc_tilt_angle(traj, 
                traj.topology, lipid_tails, blocked=False)
        _, _, apt_list = bilayer_analysis_functions.calc_APT(traj, apl_list, 
                angle_list, n_tails_per_lipid, blocked=False)
        np.savetxt('apt_permeation.dat', apt_list)
    else:
        apt_list = np.loadtxt("apt_permeation.dat")

    return apt_list

def prepare_traj(trajname='combined_nopbc.xtc'):
    """ Stitch traj files together, unwrap """
    allxtc =  glob.glob('Stage*.xtc')
    alltpr = glob.glob('Stage*.tpr')
    allgro = glob.glob('Stage*.gro')

    if not os.path.isfile('combined.xtc'):
        cmd = 'echo 0 0 | gmx trjcat -cat -f {0} {1} {2} {3} -o combined.xtc'.format(*allxtc)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE)
        p.wait()

    if not os.path.isfile(trajname):
        cmd = 'echo 0 0 | gmx trjconv -f combined.xtc -pbc mol -s {0} -o {1}'.format(alltpr[-1],
                trajname)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
        p.wait()

    return trajname, allgro[-1]

def prune_arrays(values):
    """ Combine list of arrays into a single array
    but sometimes the substituent arrays aren't the same length, so we need 
    to prune """
    smallest_length = min([len(b) for b in values])
    pruned_array = np.stack([val[:smallest_length] for val in values])
    return pruned_array

if __name__ == "__main__":
    main()
