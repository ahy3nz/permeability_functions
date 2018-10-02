import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import mdtraj
import parmed.unit as u
import pdb
import permeability_functions.misc as misc
import plot_ay
plot_ay.setDefaults()
#matplotlib.rcParams['axes.labelsize']=24
#matplotlib.rcParams['ytick.labelsize']=20
#matplotlib.rcParams['xtick.labelsize']=20

# So far this just looks at the absolute coordiante systems
ylim = [1e-8, 1e-2]
felim = [0,12]

traj = mdtraj.load('centered.gro')
midplane = traj.unitcell_lengths[0,2]/2
phosphorus_atoms = traj.topology.select('resname DSPC and name P')

top_interface_atoms = [a for a in phosphorus_atoms if traj.xyz[0,a,2] > midplane]
bot_interface_atoms = [a for a in phosphorus_atoms if traj.xyz[0,a,2] < midplane]

top_interface = np.nanmean(traj.xyz[0, top_interface_atoms,2])
bot_interface = np.nanmean(traj.xyz[0, bot_interface_atoms,2])


#all_sweeps = [thing for thing in os.listdir() if os.path.isdir(thing) and 'sweep2' not in thing and 'sweep8' not in thing and 'sweep6' not in thing and '__pycache__' not in thing]
all_sweeps = [thing for thing in os.listdir() if os.path.isdir(thing) and '__pycache__' not in thing]
all_fe_profiles = []
all_diff_profiles = []
rxn_coordinates = np.loadtxt('z_windows.out')
for sweep in all_sweeps:
    if os.path.isfile('{}/resistance_profile.dat'.format(sweep)):
        fe_profile = np.loadtxt('{}/free_energy_profile.dat'.format(sweep))[:,1]
        diffusion_profile = np.loadtxt('{}/diffusion_profile.dat'.format(sweep))[:,1]
        if (diffusion_profile[0]) > 1:
            diffusion_profile *= (u.nanometer**2)/u.second
        else:
            diffusion_profile *= (u.centimeter**2)/u.second
        diff_profile = diffusion_profile.in_units_of(u.centimeter**2/u.second)
        diff_profile = diff_profile._value
        all_fe_profiles.append(fe_profile)
        all_diff_profiles.append(diff_profile)
all_fe_profiles = np.asarray(all_fe_profiles)
all_diff_profiles = np.asarray(all_diff_profiles)

avg_fe_profile = np.nanmean(all_fe_profiles, axis=0)
avg_fe_err_profile = np.nanstd(all_fe_profiles, axis=0)/np.sqrt(all_fe_profiles.shape[0])
avg_diff_profile = np.nanmean(all_diff_profiles, axis=0)
avg_diff_err_profile = np.nanstd(all_diff_profiles, axis=0)/np.sqrt(all_diff_profiles.shape[0])
avg_fe_profile, _  = misc.symmetrize(avg_fe_profile, zero_boundary_condition=True)
avg_fe_err_profile, _ = misc.symmetrize(avg_fe_err_profile)
avg_diff_profile, _ = misc.symmetrize(avg_diff_profile)
avg_diff_err_profile, _ = misc.symmetrize(avg_diff_err_profile)

np.savetxt('avg_free_energy_profile.dat', np.column_stack((rxn_coordinates,
                                                            avg_fe_profile,
                                                            avg_fe_err_profile)))
np.savetxt('avg_diff_profile.dat', np.column_stack((rxn_coordinates,
                                                            avg_diff_profile,
                                                            avg_diff_err_profile)))

fig, ax = plt.subplots(2,1)
ax[0].plot(rxn_coordinates, avg_fe_profile)
ax[0].fill_between(rxn_coordinates, avg_fe_profile - avg_fe_err_profile,
                                    avg_fe_profile + avg_fe_err_profile,
                                    alpha=0.4)
ax[1].semilogy(rxn_coordinates, avg_diff_profile)
ax[1].fill_between(rxn_coordinates, avg_diff_profile - avg_diff_err_profile,
                                    avg_diff_profile + avg_diff_err_profile,
                                    alpha=0.4)
ax[1].axhline(y=5.87e-5, color='r', linestyle='--')
ax[1].set_ylim(ylim)

ax[0].set_ylabel(r"$\Delta$G (kcal/mol)")
ax[0].set_xlabel(r"Reaction Coordinate (nm)")

ax[1].set_ylabel(r"Diffusion (cm$^2$/sec)")
ax[1].set_xlabel(r"Reaction Coordinate (nm)")

fig.tight_layout()
fig.savefig('avg_profiles.png',transparent=True)

#########
# Now plot bootstrapping
########
n_sweeps = all_fe_profiles.shape[0]
n_bs = 1000
bootstrap_fe_profiles = []
bootstrap_diff_profiles = []
bootstrap_fe_err_profiles = []
bootstrap_diff_err_profiles = []

for _ in range(n_bs):
    bootstrap_indices = np.random.randint(0, n_sweeps, size=n_sweeps)
    bootstrap_fe_sample = []
    bootstrap_diff_sample = []
    for index in bootstrap_indices:
        bootstrap_fe_sample.append(all_fe_profiles[index,:])
        bootstrap_diff_sample.append(all_diff_profiles[index,:])
    bootstrap_fe_profiles.append(np.nanmean(bootstrap_fe_sample,axis=0))
    bootstrap_diff_profiles.append(np.nanmean(bootstrap_diff_sample,axis=0))

bootstrap_fe_profiles = np.asarray(bootstrap_fe_profiles)
bootstrap_diff_profiles = np.asarray(bootstrap_diff_profiles)

bootstrap_fe_profile = np.nanmean(bootstrap_fe_profiles, axis=0)
bootstrap_diff_profile = np.nanmean(bootstrap_diff_profiles, axis=0)

bootstrap_fe_err_profile = np.nanstd(bootstrap_fe_profiles, axis=0)/np.sqrt(n_bs)
bootstrap_diff_err_profile = np.nanstd(bootstrap_diff_profiles, axis=0)/np.sqrt(n_bs)

bootstrap_fe_profile, _ = misc.symmetrize(bootstrap_fe_profile, 
                                        zero_boundary_condition=True)
bootstrap_diff_profile, _ = misc.symmetrize(bootstrap_diff_profile)
bootstrap_fe_err_profile, _ = misc.symmetrize(bootstrap_fe_err_profile)
bootstrap_diff_err_profile, _ = misc.symmetrize(bootstrap_diff_err_profile)

fig, ax = plt.subplots(2,1)
ax[0].plot(rxn_coordinates, avg_fe_profile)
ax[0].fill_between(rxn_coordinates, bootstrap_fe_profile - bootstrap_fe_err_profile,
                                    bootstrap_fe_profile + bootstrap_fe_err_profile,
                                    alpha=0.4)
ax[1].semilogy(rxn_coordinates, bootstrap_diff_profile)
ax[1].fill_between(rxn_coordinates, bootstrap_diff_profile - bootstrap_diff_err_profile,
                                    bootstrap_diff_profile + bootstrap_diff_err_profile,
                                    alpha=0.4)
ax[1].axhline(y=5.87e-5, color='r', linestyle='--')
ax[1].set_ylim(ylim)

ax[0].set_ylabel(r"$\Delta$G (kcal/mol)")
ax[0].set_xlabel(r"Reaction Coordinate (nm)")

ax[1].set_ylabel(r"Diffusion (cm$^2$/sec)")
ax[1].set_xlabel(r"Reaction Coordinate (nm)")

fig.tight_layout()
fig.savefig('bootstrap_profiles.png',transparent=True)


#########
# Now plot log bootstrapping
########
n_sweeps = all_fe_profiles.shape[0]
n_bs = 1000
bootstrap_fe_profiles = []
bootstrap_diff_profiles = []
bootstrap_fe_err_profiles = []
bootstrap_diff_err_profiles = []

for _ in range(n_bs):
    bootstrap_indices = np.random.randint(0, n_sweeps, size=n_sweeps)
    bootstrap_fe_sample = []
    bootstrap_diff_sample = []
    for index in bootstrap_indices:
        bootstrap_fe_sample.append(all_fe_profiles[index,:])
        to_add = []
        for val in all_diff_profiles[index, :]:
            to_add.append(np.log((val)))
        bootstrap_diff_sample.append(to_add)
        #bootstrap_diff_sample.append(np.log(all_diff_profiles[index,:]))
    bootstrap_fe_profiles.append(np.nanmean(bootstrap_fe_sample,axis=0))
    bootstrap_diff_profiles.append(np.nanmean(bootstrap_diff_sample,axis=0))

bootstrap_fe_profiles = np.asarray(bootstrap_fe_profiles)
bootstrap_diff_profiles = np.asarray(bootstrap_diff_profiles)

bootstrap_fe_profile = np.nanmean(bootstrap_fe_profiles, axis=0)
bootstrap_diff_profile = np.nanmean(bootstrap_diff_profiles, axis=0)
bootstrap_diff_profile = np.exp(bootstrap_diff_profile)

bootstrap_fe_err_profile = np.nanstd(bootstrap_fe_profiles, axis=0)/np.sqrt(n_bs)
bootstrap_diff_err_profile = np.nanstd(np.exp(bootstrap_diff_profiles), axis=0)/np.sqrt(n_bs)
#bootstrap_diff_err_profile = np.exp(bootstrap_diff_err_profile)

bootstrap_fe_profile, _ = misc.symmetrize(bootstrap_fe_profile, 
                                    zero_boundary_condition=True)
bootstrap_fe_err_profile, _ = misc.symmetrize(bootstrap_fe_err_profile)
bootstrap_diff_profile, _ = misc.symmetrize(bootstrap_diff_profile)
bootstrap_diff_err_profile,_ = misc.symmetrize(bootstrap_diff_err_profile)

fig, ax = plt.subplots(2,1)
ax[0].plot(rxn_coordinates, avg_fe_profile)
ax[0].fill_between(rxn_coordinates, bootstrap_fe_profile - bootstrap_fe_err_profile, bootstrap_fe_profile + bootstrap_fe_err_profile,
                                    alpha=0.4)
ax[1].semilogy(rxn_coordinates, bootstrap_diff_profile)
ax[1].fill_between(rxn_coordinates, bootstrap_diff_profile - bootstrap_diff_err_profile,
                                    bootstrap_diff_profile + bootstrap_diff_err_profile,
                                    alpha=0.4)
ax[1].axhline(y=5.87e-5, color='r', linestyle='--')
ax[1].set_ylim(ylim)

ax[0].set_ylabel(r"$\Delta$G (kcal/mol)")
ax[0].set_xlabel(r"Reaction Coordinate (nm)")

ax[1].set_ylabel(r"Diffusion (cm$^2$/sec)")
ax[1].set_xlabel(r"Reaction Coordinate (nm)")

fig.tight_layout()
fig.savefig('log_bootstrap_profiles.png',transparent=True)

fig, ax = plt.subplots(1,1)
ax.plot(rxn_coordinates, avg_fe_profile)
ax.fill_between(rxn_coordinates, bootstrap_fe_profile - bootstrap_fe_err_profile,
                                 bootstrap_fe_profile + bootstrap_fe_err_profile,
                                 alpha=0.4)
ax.set_ylabel(r"$\Delta$G (kcal/mol)")
ax.set_xlabel(r"Reaction Coordinate (nm)")
fig.tight_layout()
fig.savefig('bootstrap_dg_profile.png',transparent=True)

fig, ax = plt.subplots(1,1)
ax.semilogy(rxn_coordinates, bootstrap_diff_profile)
ax.fill_between(rxn_coordinates, bootstrap_diff_profile - bootstrap_diff_err_profile,
                                 bootstrap_diff_profile + bootstrap_diff_err_profile,
                                 alpha=0.4)
ax.axhline(y=5.87e-5, color='r', linestyle='--')
ax.set_ylim(ylim)

ax.set_ylabel(r"Diffusion (cm$^2$/sec)")
ax.set_xlabel(r"Reaction Coordinate (nm)")
fig.tight_layout()
fig.savefig('log_bootstrap_diff_profile.png',transparent=True)


fig, ax = plt.subplots(1,1)
l, = ax.plot(rxn_coordinates, avg_fe_profile)
ax.fill_between(rxn_coordinates, bootstrap_fe_profile - bootstrap_fe_err_profile,
                                 bootstrap_fe_profile + bootstrap_fe_err_profile,
                                 alpha=0.4)
ax.set_ylabel(r"$\Delta$G (kcal/mol)", color = l.get_color())
ax.set_xlabel(r"Reaction Coordinate (nm)")
ax.set_ylim(felim)
for ytick in ax.get_yticklabels():
    ytick.set_color(l.get_color())

ax2 = ax.twinx()
second_color = '#ff7f0e'
ax2.semilogy(rxn_coordinates, bootstrap_diff_profile, color=second_color)
ax2.fill_between(rxn_coordinates, bootstrap_diff_profile - bootstrap_diff_err_profile,
                                 bootstrap_diff_profile + bootstrap_diff_err_profile,
                                 alpha=0.4, color=second_color)
ax2.axhline(y=5.87e-5, color='r', linestyle='--')
ax2.set_ylim(ylim)
for ytick in ax2.get_yticklabels():
    ytick.set_color(second_color)

ax.axvline(x=top_interface, color='k', linestyle=':')
ax.axvline(x=bot_interface, color='k', linestyle=':')
ax2.set_ylabel(r"Diffusion (cm$^2$/sec)", color=second_color)
plot_ay.tidyUp(fig, ax, tightLayoutArgs={}, gridArgs={'axis':'y'})
fig.tight_layout()
fig.savefig('stacked_profiles.png',transparent=True)
