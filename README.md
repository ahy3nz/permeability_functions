# permeability_functions
Functions for computing permeability from timeseries of forces

* `thermo_functions.py` (module) has most of the functions used for computing
free energies, diffusion coefficients, resistances, and permeability values.
Note the use of the `simtk.unit` package to carry units throughout
the various calculations

* `misc.py` (module) has some utility functions for doing these calculations

* `grid_functions.py` (module) has some functions for analyzing non-flat interfaces

* `scripts/absolute_analysis.py` (script) is the code used to analyze a set of 
permeability sweeps and simulations, generating the various profiles

* `scripts/bootstrap.py` (script) is the code used to log-bootstrap from
the already-gathered permeability data

* `scripts/read_profiles.py` (script) is the code used to analyze the profiles
from each sweep, calculate a permeability coefficient, and dump to `csv`

* `scripts/plot_overlay_profiles.py` (convenience script) is used to help generate
nice profiles that can be laid on top of simulation renderings of bilayer 
structures

* `scripts/plot_profiles.py` (script) generates bootstrapped-profile-images for
free energy, diffusion, and resistance

* `scripts/analyze_water_disordering.py` (script) analyzes bilayers properties
over the course of restraining/constraining simulations

* `scripts/relative_analysis.py` (script) is the code used to analyze a 
set of permeability sweeps and simulations, but trying to account for
uneven interfaces
