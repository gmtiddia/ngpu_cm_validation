from val_config import configuration
from val_helpers import __get_distributions, __get_distributions_csv, __plot_distributions, __get_emd, __get_emd_csv, __plot_emd, __get_ks_test, __get_ks_csv, __plot_ks

# simulation sets we want to get the distributions from
sim_set = ['Sim 1', 'Sim 2', 'Sim 3']
__get_distributions(set = sim_set)

if(configuration['plot_distributions']):
    # chose the set of NEST simulations to compare with NEST GPU ('NEST 1' or 'NEST 2')
    __get_distributions_csv(simulation = 'Sim 1')
    # chose the dataX set to plot
    __plot_distributions(run_id=0)

if(configuration['emd']):
    __get_emd()
    if(configuration['emd_boxplots']):
        __get_emd_csv()
        __plot_emd()

if(configuration['ks']):
    __get_ks_test()
    if(configuration['ks_boxplots']):
        __get_ks_csv()
        __plot_ks()









