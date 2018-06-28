import simtk.unit as u
def validate_array_type(array, desired_unit):
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
    if isinstance(array._value[0], u.Quantity):
        array= array.unit*np.array([d._value for d in array._value])
    return array


