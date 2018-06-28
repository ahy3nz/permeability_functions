import itertools
import numpy as np
import mdtraj
import permeability as prm
import grid_analysis
import bilayer_analysis_functions
import simtk.unit as u

# 1) Compute means and force correlations (analyze_sweeps)
# 2) Integrate force correlations  (integrate_acf_over_time)
# 3) For each window, recover the tracer and get the distances from interface
# 4) Integrate mean forces and force autocorrelations based on these distances

def analyze_force_timeseries(times, forces, meanf_name, fcorr_name,
                            correlation_length=300*u.picosecond):
    """ Given a timeseries of forces, compute force autocorrealtions and means"""
    mean_force = np.mean(forces)
    dstep = ((times[1] - times[0]) * u.femtosecond).in_units_of(u.picosecond)
    funlen = int(correlation_length/dstep)._value
    FACF = prm.acf(forces, None, funlen, dstart=10)
    time_intervals = np.arange(0, funlen*dstep._value, dstep._value )
    times_facf = np.column_stack((times,FACF))
    np.savetxt(fcorr_name, times_facf)
    np.savetxt(meanf_name, [mean_force])

    return mean_force, times_facf

def integrate_facf_over_time(times, facf, average_fraction=0.1):
    """ Integrate force autocorelations

    Notes
    -----
    We're doing a cumulative sum, but average the 'last bit' in order 
    to average out the noise
    """
    intF = np.cumsum(facf)*(times[1]-times[0])
    lastbit = int((1.0-average_fraction)*intF.shape[0])
    intFval = np.mean(intF[-lastbit:])

    return intF, intFval

def compute_free_energy_profile(forces, distances):
    if not isinstance(forces, u.Quantity):
        forces = forces * (u.kilocalorie / (u.mole * u.angstrom))
    else:
        forces = forces.in_units_of( (u.kilocalorie/ (u.mole*u.angstrom)))

    if not isinstance(distances, u.Quantity):
        distances = distances * u.angstrom
    else:
        distances = distances.in_units_of(u.angstrom)

    return -scipy.integrate.cumtrapz(forces, x=distances, initial=0)

def compute_diffusion_constant(intfacf, 
                                kb=1.987e-3 * u.kilocalorie / (u.mole * u.kelvin),
                                temp=305*u.kelvin):
    if not isinstance(intfacf, u.Quantity):
        intfacf = (intfacf 
                * (u.kilocalorie / (u.mole * u.angstrom))**2 * u.picosecond)
    else:
        intfacf = intfacf.in_units_of((
                        u.kilocalorie / (u.mole * u.angstrom))**2 * u.picosecond)

    RT2 = (kb*temp)**2
    diffusion_constant = (RT2/intfacf).in_units_of(u.nanometer**2/u.second)

    return diffusion_constant

def compute_resistance_profile(fe_profile, diff_profile, distances,
                                kb=1.987e-3 * u.kilocalorie / (u.mole * u.kelvin),
                                temp=305*u.kelvin):
    if not isinstance(fe_profile, u.Quantity):
        fe_profile = fe_profile * u.kilocalorie/u.mole 
    else:
        fe_profile = fe_profile.in_units_of(u.kilocalorie/u.mole)

    if not isinstance(diff_profile, u.Quantity):
        diff_profile = diff_profile * u.nanometer**2/u.second
    else:
        diff_profile = diff_profile.in_units_of(u.nanometer**2/u.second)

    numerator = np.exp(fe_profile/(kb*temp))
    return scipy.integrate.cumtrapz(numerator/diff_profile, x=distances, initial=0)

def compute_permeability_profile(resistance_profile):
    return 1/resistance_profile


