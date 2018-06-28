import numpy as np
import simtk.unit as u
def validate_quantity_type(array, desired_unit):
    """ Ensure that arrays are u.Quantity, but elements are just floats
    Parameters
    ---------
    array : some iterable
    desired_unit : simtk.Unit

    Returns
    -------
    array : array in correct form
    """
    if isinstance(array, u.Quantity):
        array = array.in_units_of(desired_unit)
    else:
        array = array * desired_unit 
    try:
        if isinstance(array._value[0], u.Quantity):
            array= array.unit*np.array([d._value for d in array._value])
    except IndexError:
        pass

    return array

def symmetrize(data, zero_boundary_condition=False):
    """Symmetrize a profile
    
    Params
    ------
    data : np.ndarray, shape=(n,)
        Data to be symmetrized
    zero_boundary_condition : bool, default=False
        If True, shift the right half of the curve before symmetrizing

    Returns
    -------
    dataSym : np.ndarray, shape=(n,)
        symmetrized data
    dataSym_err : np.ndarray, shape=(n,)
        error estimate in symmetrized data

    This function symmetrizes a 1D array. It also provides an error estimate
    for each value, taken as the standard error between the "left" and "right"
    values. The zero_boundary_condition shifts the "right" half of the curve 
    such that the final value goes to 0. This should be used if the data is 
    expected to approach zero, e.g., in the case of pulling a water molecule 
    through one phase into bulk water.
    """
    n_windows = data.shape[0]
    n_win_half = int(np.ceil(float(n_windows)/2))
    dataSym = np.zeros_like(data)
    dataSym_err = np.zeros_like(data)
    shift = {True: data[-1], False: 0.0}
    for i, sym_val in enumerate(dataSym[:n_win_half]):
        val = 0.5 * (data[i] + data[-(i+1)])
        err = np.std([data[i], data[-(i+1)] - shift[zero_boundary_condition]]) / np.sqrt(2)
        dataSym[i], dataSym_err[i] = val, err
        dataSym[-(i+1)], dataSym_err[-(i+1)] = val, err        
    if zero_boundary_condition:
        dataSym[:] -= dataSym[0]
    return dataSym, dataSym_err

