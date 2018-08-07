import numpy as np
import pandas as pd
import pdb
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

df = pd.read_csv('permeability_summary.csv')
#df = pd.read_csv('d_from_leaflet_permeability.csv')
#df = pd.read_csv('d_from_local_permeability.csv')
n_bs = 100000
matplotlib.rcParams['axes.titlesize'] = 24
matplotlib.rcParams['axes.labelsize'] = 24

regular_mean = np.mean(df['permeability'].values)
regular_error = np.std(df['permeability'].values)/np.sqrt(len(df['permeability'].values))
bootstrap = np.random.choice(df['permeability'].values, 
        size=(n_bs, len(df['permeability'].values)))
bootstrap_distribution = np.mean(bootstrap, axis=1)
bootstrap_mean = np.mean(bootstrap_distribution)
bootstrap_error = np.std(bootstrap_distribution)/np.sqrt(n_bs)

log_data = np.log(df['permeability'].values)
new_bootstrap = np.random.choice(log_data, size=(n_bs, len(log_data)))
new_bootstrap_distribution = np.mean(new_bootstrap, axis=1)




fig, ax = plt.subplots(1,1)
ax.hist(new_bootstrap_distribution, density=True)
ax.set_xlabel("log(permeability(cm/sec))")
ax.set_title("{} bootstrap samples".format(n_bs))
fig.tight_layout()
fig.savefig('bootstrap_logperm_distribution.png', transparent=True)
plt.close(fig)

new_bootstrap_mean = np.mean(new_bootstrap_distribution)
bootstrap_log_error = np.std(np.exp(new_bootstrap_distribution))/np.sqrt(n_bs)
log_mean = np.exp(new_bootstrap_mean)
print("Regular mean: {0} ({3}), "
        "\nbootstrap_mean: {1} ({4}), "
        "\nbootstrap_log_mean: {2} ({5})".format(
                    regular_mean, bootstrap_mean, log_mean,
                    regular_error, bootstrap_error, bootstrap_log_error))
