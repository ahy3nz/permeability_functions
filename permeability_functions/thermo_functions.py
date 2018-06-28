import scipy.integrate
import numpy as np
import mdtraj
import grid_analysis
import bilayer_analysis_functions
import simtk.unit as u
import permeability_functions.misc as misc

# 1) Compute means and force correlations (analyze_sweeps)
# 2) Integrate force correlations  (integrate_acf_over_time)
# 3) For each window, recover the tracer and get the distances from interface
# 4) Integrate mean forces and force autocorrelations based on these distances

def analyze_force_timeseries(times, forces, meanf_name, fcorr_name,
                            correlation_length=300*u.picosecond):
    """ Given a timeseries of forces, compute force autocorrealtions and means"""
    mean_force = np.mean(forces)
    times = misc.validate_array_type(times, u.picosecond)
    dstep = times[1] - times[0]
    funlen = int(correlation_length/dstep)
    FACF = acf(forces, funlen, dstart=10)
    time_intervals = np.arange(0, funlen*dstep._value, dstep._value )*dstep.unit
    time_intevals = misc.validate_array_type(time_intervals, dstep.unit)
    times_facf = np.column_stack((time_intervals, FACF))
    np.savetxt(fcorr_name, times_facf)
    np.savetxt(meanf_name, [mean_force._value])

    return mean_force, time_intervals, FACF

def acf(forces, funlen, dstart=10):
    """Calculate the autocorrelation of a function

    Params
    ------
    forces : np.ndarray, shape=(n,)
        The force timeseries acting on a molecules
    timestep : float
        Simulation timestep in fs
    funlen : int
        The desired length of the correlation function

    Returns
    -------
    corr : np.array, shape=(funlen,)
        The autocorrelation of the forces
    """    
    if funlen > forces.shape[0]:
       raise Exception("Not enough data")
    # number of time origins
    ntraj = int(np.floor((forces.shape[0]-funlen)/dstart))
    meanfz = np.mean(forces)
    f1 = np.zeros((funlen)) * meanfz.unit**2
    origin = 0 
    for i in range(ntraj):
        dfzt = (forces[origin:origin+funlen] - meanfz)
        dfz0 = (forces[origin] - meanfz)
        f1 += dfzt*dfz0
        origin += dstart
    return f1/ntraj


def integrate_facf_over_time(times, facf, average_fraction=0.1):
    """ Integrate force autocorelations

    Notes
    -----
    We're doing a cumulative sum, but average the 'last bit' in order 
    to average out the noise
    """
    intF = np.cumsum(facf)*facf.unit * (times[1]-times[0])
    lastbit = int((1.0-average_fraction)*intF.shape[0])
    intFval = np.mean(intF[-lastbit:])

    return intF, intFval

def compute_free_energy_profile(forces, distances):
    """
    forces : array of floats, u.Quantity
    distances: array of floats, u.Quantity

    Notes
    -----
    Forces and distances are u.Quantity, but the elements should just be floats
    """
    forces = misc.validate_array_type(forces, (u.kilocalorie / (u.mole * u.angstrom)))
    distances = misc.validate_array_type(distances, u.angstrom)

    return -scipy.integrate.cumtrapz(forces._value, x=distances._value, initial=0)*forces.unit*distances.unit

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


