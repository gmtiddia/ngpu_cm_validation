from val_config import configuration
from val_helpers import __get_distributions, __get_distributions_csv, __plot_distributions, __get_emd, __get_emd_csv, __plot_emd

# simulation sets we want to get the distributions from
sim_set = ['NEST 1', 'NEST 2', 'NEST GPU']

__get_distributions(set = sim_set)


if(configuration['plot_distributions']):
    # chose the set of NEST simulations to compare with NEST GPU ('NEST 1' or 'NEST 2')
    __get_distributions_csv(nest_simulation = 'NEST 1')
    # chose the dataX set to plot
    __plot_distributions(run_id=0)


if(configuration['emd']):
    __get_emd()
    if(configuration['emd_boxplots']):
        __get_emd_csv()
        __plot_emd()









