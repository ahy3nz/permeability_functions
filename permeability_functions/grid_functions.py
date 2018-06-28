import itertools
import numpy as np
import mdtraj
import permeability as prm
import grid_analysis
import bilayer_analysis_functions
import simtk.unit as u

def distance_from_interface(traj, tracer_resid):
    """ Given a trajectory and a tracer residue, find the closest interface

    Note
    -----
    Comparisons are based on time-averaged interfaces and coordinates
    """
    res = traj.topology.residue(tracer_resid-1)
    tracer_oxygen = res.atom(0)
    water_indices = traj.topology.select('water') 
    headgroup_indices = grid_analysis._get_headgroup_indices(traj)

    # Get leaflet interfaces
    bot_interface, top_interface  = find_interface_lipid(traj, headgroup_indices)
    leaflet_interfaces = [np.mean(bot_interface), np.mean(top_interface)]


    # Find local interface within each grid
    _, xbin_centers, ybin_centers, xedges, yedges = grid_surface(traj, grid_size=1.0)
    xbin_width = xbin_centers[1] - xbin_centers[0]
    ybin_width = ybin_centers[1] - ybin_centers[0]

    top_interface_grid = np.zeros((len(xbin_centers), len(ybin_centers)))
    bot_interface_grid = np.zeros((len(xbin_centers), len(ybin_centers)))
    for i, x in enumerate(xbin_centers):
        for j, y in enumerate(ybin_centers):
            atoms_xy = grid_analysis._find_atoms_within(traj, x=x, y=y, 
                    atom_indices=headgroup_indices,
                    xbin_width=xbin_width, ybin_width=ybin_width)
            z_interface_bot, z_interface_top = find_interface_lipid(traj, atoms_xy)
            (bot_interface_avg, top_interface_avg) = (np.mean(z_interface_bot), 
                                                    np.mean(z_interface_top))
            bot_interface_grid[i,j] = bot_interface_avg
            top_interface_grid[i,j] = top_interface_avg

    
    # Identify which xy region we're in 
    xyz = np.mean(traj.xyz[:, tracer_oxygen.index, :], axis=0)
    bin_x = np.digitize(xyz[0], xedges) - 1
    bin_y = np.digitize(xyz[1], yedges) - 1

    # Identify the z interface for this xy region
    (interface_bot, interface_top) = (bot_interface_grid[bin_x, bin_y],
                                        top_interface_grid[bin_x, bin_y])

    # Find distance from local interface and leaflet interface
    # Pick the closer interface
    if abs(interface_bot - xyz[2]) < abs(interface_top - xyz[2]):
        d_from_local_i = interface_bot - xyz[2]
        d_from_leaflet_i = leaflet_interfaces[0] - xyz[2]
        closest_interface = interface_bot
    else:
        d_from_local_i = xyz[2] - interface_top
        d_from_leaflet_i = xyz[2] - leaflet_interfaces[1] 
        closest_interface = interface_top

    return d_from_local_i, d_from_leaflet_i, closest_interface
    
def find_interface_lipid(traj, headgroup_indices):
    """ Find the interface based on lipid head groups"""

    # Sort into top and bottom leaflet
    #midplane = np.mean(traj.xyz[:,headgroup_indices,2])
    midplane = np.mean(traj.unitcell_lengths[:,2])/2
    bot_leaflet = [a for a in headgroup_indices if traj.xyz[0,a,2] < midplane and abs(traj.xyz[0,a,2] - midplane) > 1]
    top_leaflet = [a for a in headgroup_indices if traj.xyz[0,a,2] > midplane and abs(traj.xyz[0,a,2] - midplane) > 1]

    com_bot = mdtraj.compute_center_of_mass(traj.atom_slice(bot_leaflet))
    com_top = mdtraj.compute_center_of_mass(traj.atom_slice(top_leaflet))

    return com_bot[:,2], com_top[:,2]

def grid_surface(traj, grid_size=0.2):
    """ Compute a density heatmap by gridding up space """

    atom_indices = [a.index for a in traj.topology.atoms]
    xbounds = (np.min(traj.xyz[:, :, 0]),
            np.max(traj.xyz[:, :, 0]))
    ybounds = (np.min(traj.xyz[:, :, 1]),
            np.max(traj.xyz[:, :, 1]))
    zbounds = (np.min(traj.xyz[:, :, 2]),
            np.max(traj.xyz[:, :, 2]))

    n_xbins = int(round((xbounds[1] - xbounds[0]) / grid_size))
    xbin_width = (xbounds[1] - xbounds[0]) / n_xbins

    n_ybins = int(round((ybounds[1] - ybounds[0]) / grid_size))
    ybin_width = (ybounds[1] - ybounds[0]) / n_ybins

    thickness = zbounds[1] - zbounds[0]
    density_profile=[]
    v_slice = xbin_width * ybin_width * thickness * u.nanometer**3

    masses = (bilayer_analysis_functions.get_all_masses(traj, traj.topology, atom_indices) / v_slice).in_units_of(u.kilogram * (u.meter**-3))._value

    for i, frame in enumerate(traj):
        xyz = traj.xyz[i, :, :]
        hist, xedges, yedges = np.histogram2d(xyz[:,0], xyz[:,1],
                bins=[n_xbins, n_ybins], range=[xbounds, ybounds], 
                normed=False, weights=masses)
        density_profile.append(hist/v_slice._value)
        xbin_centers = xedges[1:] - xbin_width / 2
        ybin_centers = yedges[1:] - ybin_width / 2

    return np.array(density_profile), xbin_centers, ybin_centers, xedges, yedges

