import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import os
import sys
import elephant
from elephant.statistics import isi, cv
from elephant.conversion import BinnedSpikeTrain
from neo.core import SpikeTrain
from quantities import s
from elephant.spike_train_correlation import corrcoef
from scipy.stats import wasserstein_distance
from scipy.stats import kstest
from val_config import configuration


def __gather_metadata(path):
    """ Reads first and last ids of
    neurons in each population.

    Parameters
    ------------
    path
        Path where the spike detector files are stored.

    Returns
    -------
    node_ids
        Lowest and highest id of nodes in each population.

    """
    # load node IDs
    node_idfile = open(path + '/population_nodeids.dat', 'r')
    node_ids = []
    for l in node_idfile:
        node_ids.append(l.split())
    node_ids = np.array(node_ids, dtype='i4')
    return node_ids


def __load_spike_times(path, begin, end, npop):
    """ Loads spike times of each spike detector.

    Parameters
    ----------
    path
        Path where the files with the spike times are stored.
    begin
        Time point (in ms) to start loading spike times (included).
    end
        Time point (in ms) to stop loading spike times (included).
    npop
        Number of neuron populations.

    Returns
    -------
    data
        Dictionary containing spike times in the interval from ``begin``
        to ``end``.

    """
    node_ids = __gather_metadata(path)
    data = {}
    dtype = {'names': ('sender', 'time_ms'),  # as in header
             'formats': ('i4', 'f8')}
    #print(node_ids)

    sd_names = {}
    
    for i_pop in range(npop):
        fn = os.path.join(path, 'spike_times_' + str(i_pop) + '.dat')
        data_i_raw = np.loadtxt(fn, skiprows=1, dtype=dtype)

        data_i_raw = np.sort(data_i_raw, order='time_ms')
        # begin and end are included if they exist
        low = np.searchsorted(data_i_raw['time_ms'], v=begin, side='left')
        high = np.searchsorted(data_i_raw['time_ms'], v=end, side='right')
        data[i_pop] = data_i_raw[low:high]
        sd_names[i_pop] = 'spike_times_' + str(i_pop)

    spike_times_list = []
    for i, n in enumerate(sd_names):
        spike_times = []
        for id in np.arange(node_ids[i, 0], node_ids[i, 1] + 1):
            spike_times.append([])
        for row in data[i]:
            sender = row[0]
            time = row[1]/1000.0
            i_neur = sender - node_ids[i, 0]
            spike_times[i_neur].append(time)
            
        spike_times_list.append(spike_times)

    return spike_times_list


def __get_distributions(set = ['Sim 1', 'Sim 2', 'Sim 3']):
    nrun = configuration['nrun']
    npop = 8
    begin = 500.0
    end = 10500.0
    matrix_size = 200
    spike_time_bin = 0.002

    for sim in set:
        print(sim+' spike data processing')

        for i_run in range(nrun):
            if(sim=='Sim 1'):
                path = configuration['sim1_path']+'data'+str(i_run)
            if(sim=='Sim 2'):
                path = configuration['sim2_path']+'data'+str(i_run)
            if(sim=='Sim 3'):
                path = configuration['sim3_path']+'data'+str(i_run)
            
            dum = 0
            for i in range(npop):
                for j in range(len(configuration['distributions'])):
                    if(os.path.isfile(path+"/"+configuration['distributions'][j]+"_"+str(i)+".dat")==False):
                        dum += 1
            if(dum==0):
                #print("Distributions already computed for run "+str(i_run+1))
                pass
            else:
                print ('Processing dataset '+ str(i_run+1) + '/' + str(nrun))
                spike_times_list = __load_spike_times(path, begin, end, npop)

                for ipop in range(npop):
                    print("Calculating distributions for population:", ipop+1, flush=True)
                    spike_times = spike_times_list[ipop]

                    dist = []
                    # computing firing rate
                    if('firing_rate' in configuration['distributions']):
                        print("Computing firing rate", flush=True)
                        for st_row in spike_times:
                            if len(st_row)==0:
                                dist.append(0.0)
                            else:
                                dist.append(elephant.statistics.mean_firing_rate(np.array(st_row),
                                                                            begin/1000.0,
                                                                            end/1000.0))
                        file_data = open(path+"/firing_rate_"+str(ipop)+".dat", "w")
                        np.savetxt(file_data, dist)
                        file_data.close()
                        dist = []

                    # computing CV ISI
                    if('cv_isi' in configuration['distributions']):
                        print("Computing CV ISI", flush=True)
                        for st_row in spike_times:
                            if (len(st_row) > 1):
                                dist.append(cv(isi(np.array(st_row))))


                        file_data = open(path+"/cv_isi_"+str(ipop)+".dat", "w")
                        np.savetxt(file_data, dist)
                        file_data.close()
                        dist = []

                    # computing correlation
                    if('correlation' in configuration['distributions']):
                        print("Computing correlation", flush=True)
                        st_list = []
                        for j in range(matrix_size):
                            spike_train = SpikeTrain(np.array(spike_times[j])*s,
                                                    t_stop = (end/1000.0)*s)
                            st_list.append(spike_train)

                        binned_st = BinnedSpikeTrain(st_list, spike_time_bin*s, None,
                                                    (begin/1000.0)*s, (end/1000.0)*s, tolerance=None)

                        cc_matrix = corrcoef(binned_st)
                        for j in range(matrix_size):
                            for k in range(matrix_size):
                                    #print(j, k, cc_matrix[j][k])                                                                                       
                                    if (j != k and not np.isnan(cc_matrix[j][k])):
                                        dist.append(cc_matrix[j][k])
                        
                        if(len(dist)>0):
                            np.savetxt(path+"/correlation_"+str(ipop)+".dat", dist)
                        else:
                            np.savetxt(path+"/correlation_"+str(ipop)+".dat", [])
                        dist = []


def __get_distributions_csv(simulation = 'Sim 1'):
    nrun = configuration['nrun']
    npop = 8
    dum = 0
    for i_run in range(nrun):
        for d in configuration['distributions']:
            if(os.path.isfile("csv/"+d+"_"+str(i_run)+".csv")==False):
                dum += 1
    if dum>0:
        for i_run in range(nrun):
            print ('Loading dataset '+ str(i_run+1) + '/' + str(nrun), flush=True)
            if(simulation=='Sim 1'):
                path1 = configuration['sim1_path']+ 'data' + str(i_run)
            else:
                path1 = configuration['sim2_path']+ 'data' + str(i_run)
            path2 = configuration['sim3_path']+ 'data' + str(i_run)
            for d in configuration['distributions']:
                
                    print(d+' distribution')
                    distlist = []                                                                                                                                                                                                                           
                    popid = []
                    sim = []
                    for ipop in range(npop):
                        print ("run ", i_run, "/", nrun, "  pop ", ipop, "/", npop, flush=True)
                        d1 = np.loadtxt(path1+"/"+d+"_"+str(ipop)+".dat")
                        d2 = np.loadtxt(path2+"/"+d+"_"+str(ipop)+".dat")
                        distlist += [i for i in d1]
                        distlist += [i for i in d2]
                        sim += ["NEST GPU main" for i in range(len(d1))]
                        sim += ["NEST GPU conn" for i in range(len(d2))]
                        popid += [ipop for i in range(len(d1))]
                        popid += [ipop for i in range(len(d2))]
                    dataset = {d: distlist, "popid": popid, "Simulator": sim}
                    data = pd.DataFrame(dataset)
                    data.to_csv("csv/"+d+"_"+str(i_run)+".csv", index=False)


def __plot_distributions(run_id=0):
    print("Plotting distributions")
    cifre=25
    titolo=25
    layer=['L2/3E', 'L2/3I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
    #colore="Set2"
    colors = ['#fc6333', '#33BBEE']
    sns.set_palette(sns.color_palette(colors))

    fig=plt.figure(figsize=(32, 18), dpi=300)
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.5)

    #plt.suptitle("Cortical microcircuit model distributions", fontsize = titolo + 3)

    print("Loading Firing Rate")
    #firing_rate = get_all(pd.read_csv("firing_rate/firing_rate.csv"), "fr", 0.0, 100.0)
    firing_rate = pd.read_csv("csv/firing_rate_"+str(run_id)+".csv")

    ax1 = plt.subplot(gs[0, :2])
    plt.text(-0.125,1.03,"A", transform=ax1.transAxes, weight="bold", fontsize=titolo)
    if(configuration['distribution_visual']=='boxplot'):
        v1 = sns.boxplot(x="popid", y="firing_rate", hue="Simulator", data=firing_rate)
    elif(configuration['distribution_visual']=='violinplot'):
        v1 = sns.violinplot(x="popid", y="firing_rate", hue="Simulator", data=firing_rate, split=True, inner="quartile", bw="silverman", gridsize=300, cut=0)
    else:
        print("Please chose a valid entry ('boxplot' or 'violinplot') for ditribution visualization.")
        sys.exit()
    for l in v1.lines:
        #l.set_linestyle('--')
        l.set_linewidth(1.5)
        l.set_color('black')
        #l.set_alpha(0.8)
    for l in v1.lines[1::3]:
        #l.set_linestyle('--')
        l.set_linewidth(1.5)
        l.set_color('black')
        #l.set_alpha(0.8)
    plt.xticks(np.arange(len(layer)), layer)
    plt.xlabel("")
    plt.ylabel("Firing rate [spikes/s]", size=titolo)
    plt.tick_params(labelsize=cifre)
    plt.grid()
    plt.legend([],[], frameon=False)

    del firing_rate
    print("Loading CV ISI")
    cv_isi = pd.read_csv("csv/cv_isi_"+str(run_id)+".csv")

    ax2 = plt.subplot(gs[0, 2:])
    plt.text(-0.125,1.03,"B", transform=ax2.transAxes, weight="bold", fontsize=titolo)
    if(configuration['distribution_visual']=='boxplot'):
        v2 = sns.boxplot(x="popid", y="cv_isi", hue="Simulator", data=cv_isi)
    if(configuration['distribution_visual']=='violinplot'):
        v2 = sns.violinplot(x="popid", y="cv_isi", hue="Simulator", data=cv_isi, split=True, inner="quartile", bw="silverman", gridsize=300, cut=0)

    for l in v2.lines:
        #l.set_linestyle('--')
        l.set_linewidth(1.5)
        l.set_color('black')
        #l.set_alpha(0.8)
    for l in v2.lines[1::3]:
        #l.set_linestyle('--')
        l.set_linewidth(1.5)
        l.set_color('black')
        #l.set_alpha(0.8)
    plt.xticks(np.arange(len(layer)), layer)
    plt.xlabel("")
    plt.ylabel("CV ISI", size=titolo)
    plt.tick_params(labelsize=cifre)
    plt.grid()
    plt.legend([],[], frameon=False)

    del cv_isi
    print("Loading Correlation")
    #correlation = get_all(pd.read_csv("correlation/correlation.csv"), "corr", -0.05, 0.2)
    correlation = pd.read_csv("csv/correlation_"+str(run_id)+".csv")

    ax3 = plt.subplot(gs[1, 1:3])
    plt.text(-0.125,1.03,"C", transform=ax3.transAxes, weight="bold", fontsize=titolo)
    if(configuration['distribution_visual']=='boxplot'):
        v3 = sns.boxplot(x="popid", y="correlation", hue="Simulator", data=correlation)
    if(configuration['distribution_visual']=='violinplot'):
        v3 = sns.violinplot(x="popid", y="correlation", hue="Simulator", data=correlation, split=True, inner="quartile", bw="silverman", gridsize=400)
    for l in v3.lines:
        #l.set_linestyle('--')
        l.set_linewidth(1.5)
        l.set_color('black')
        #l.set_alpha(0.8)
    for l in v3.lines[1::3]:
        #l.set_linestyle('--')
        l.set_linewidth(1.5)
        l.set_color('black')
        #l.set_alpha(0.8)
    plt.xticks(np.arange(len(layer)), layer)
    plt.xlabel("")
    plt.ylabel("correlation", size=titolo)
    plt.ylim(-0.05,0.12)
    plt.tick_params(labelsize=cifre)
    plt.legend(bbox_to_anchor=(1.6, 0.5),loc='center right', prop={'size': titolo})
    plt.grid()
    #plt.show()

    del correlation
    print("Plot")

    plt.subplots_adjust(top=0.9, hspace = 0.25)
    plt.savefig("distributions_"+configuration['distribution_visual']+"_"+str(run_id)+".pdf")


def __get_emd():
    nrun = configuration['nrun']
    npop = 8

    for d in configuration['distributions']:
        if(os.path.isfile("emd_"+d+"_sim1_sim3.dat")==False or os.path.isfile("emd_"+d+"_sim1_sim2.dat")==False):
            emd_sim1_sim3=np.zeros((npop,nrun))
            emd_sim1_sim2=np.zeros((npop,nrun))
            ks_sim1_sim3=np.zeros((npop,nrun))
            ks_sim1_sim2=np.zeros((npop,nrun))
            print('EMD '+d)
            for i_run in range(nrun):
                print ('Loading dataset '+ str(i_run+1) + '/' + str(nrun), flush=True)
                path1 = configuration['sim1_path'] + 'data' + str(i_run)
                path2 = configuration['sim3_path'] + 'data' + str(i_run)
                path3 = configuration['sim2_path'] + 'data' + str(i_run)
                for ipop in range(npop):
                    print ("run ", i_run, "/", nrun, "  pop ", ipop, "/", npop, flush=True)
                    dist1 = np.loadtxt(path1+"/"+d+"_"+str(ipop)+".dat")
                    dist2 = np.loadtxt(path2+"/"+d+"_"+str(ipop)+".dat")
                    dist3 = np.loadtxt(path3+"/"+d+"_"+str(ipop)+".dat")

                    emd_sim1_sim3[ipop, i_run] = wasserstein_distance(dist1,dist2)
                    emd_sim1_sim2[ipop, i_run] = wasserstein_distance(dist1,dist3)
                    

            np.savetxt("emd_"+d+"_sim1_sim3.dat", emd_sim1_sim3)
            np.savetxt("emd_"+d+"_sim1_sim2.dat", emd_sim1_sim2)


def __get_ks_test():
    nrun = configuration['nrun']
    npop = 8

    for d in configuration['distributions']:
        if(os.path.isfile("ks_"+d+"_sim1_sim3.dat")==False or os.path.isfile("ks_"+d+"_sim1_sim2.dat")==False):
            ks_sim1_sim3=np.zeros((npop,nrun))
            ks_sim1_sim2=np.zeros((npop,nrun))
            print('EMD '+d)
            for i_run in range(nrun):
                print ('Loading dataset '+ str(i_run+1) + '/' + str(nrun), flush=True)
                path1 = configuration['sim1_path'] + 'data' + str(i_run)
                path2 = configuration['sim3_path'] + 'data' + str(i_run)
                path3 = configuration['sim2_path'] + 'data' + str(i_run)
                for ipop in range(npop):
                    print ("run ", i_run, "/", nrun, "  pop ", ipop, "/", npop, flush=True)
                    dist1 = np.loadtxt(path1+"/"+d+"_"+str(ipop)+".dat")
                    dist2 = np.loadtxt(path2+"/"+d+"_"+str(ipop)+".dat")
                    dist3 = np.loadtxt(path3+"/"+d+"_"+str(ipop)+".dat")

                    ks_sim1_sim3[ipop, i_run] = kstest(dist1,dist2)[0]
                    ks_sim1_sim2[ipop, i_run] = kstest(dist1,dist3)[0]

            np.savetxt("ks_"+d+"_sim1_sim3.dat", ks_sim1_sim3)
            np.savetxt("ks_"+d+"_sim1_sim2.dat", ks_sim1_sim2)


def __get_emd_csv():
    for d in configuration['distributions']:
        print("EMD "+d+" csv")
        emd_sim1_sim3 = np.loadtxt('emd_'+d+"_sim1_sim3.dat")
        emd_sim1_sim2 = np.loadtxt('emd_'+d+"_sim1_sim2.dat")
        npop = 8
        nrun = configuration['nrun']
        emd_list = []
        popid = []
        sim = []
        for ipop in range(npop):
            dum_sim1_sim30 = emd_sim1_sim3[ipop,:]
            dum_sim1_sim3 = []
            dum_sim1_sim20 = emd_sim1_sim2[ipop,:]
            dum_sim1_sim2 = []
            for i in range(nrun):
                if(dum_sim1_sim30[i] != np.nan):
                    dum_sim1_sim3.append(dum_sim1_sim30[i])
                    emd_list.append(dum_sim1_sim30[i])
                if(dum_sim1_sim20[i] != np.nan):
                    dum_sim1_sim2.append(dum_sim1_sim20[i])
                    emd_list.append(dum_sim1_sim20[i])
            for i in range(len(dum_sim1_sim3)):
                popid.append(ipop)
                sim.append("main-conn")
            for i in range(len(dum_sim1_sim2)):
                popid.append(ipop)
                sim.append("main-main")

        dataset = {"EMD": emd_list, "popid": popid, "Simulator": sim}
        data = pd.DataFrame(dataset)
        data.to_csv('csv/emd_'+d+".csv", index=False)


def __get_ks_csv():
    for d in configuration['distributions']:
        print("KS test "+d+" csv")
        ks_sim1_sim3 = np.loadtxt('ks_'+d+"_sim1_sim3.dat")
        ks_sim1_sim2 = np.loadtxt('ks_'+d+"_sim1_sim2.dat")
        npop = 8
        nrun = configuration['nrun']
        ks_list = []
        popid = []
        sim = []
        for ipop in range(npop):
            dum_sim1_sim30 = ks_sim1_sim3[ipop,:]
            dum_sim1_sim3 = []
            dum_sim1_sim20 = ks_sim1_sim2[ipop,:]
            dum_sim1_sim2 = []
            for i in range(nrun):
                if(dum_sim1_sim30[i] != np.nan):
                    dum_sim1_sim3.append(dum_sim1_sim30[i])
                    ks_list.append(dum_sim1_sim30[i])
                if(dum_sim1_sim20[i] != np.nan):
                    dum_sim1_sim2.append(dum_sim1_sim20[i])
                    ks_list.append(dum_sim1_sim20[i])
            for i in range(len(dum_sim1_sim3)):
                popid.append(ipop)
                sim.append("main-conn")
            for i in range(len(dum_sim1_sim2)):
                popid.append(ipop)
                sim.append("main-main")

        dataset = {"KS": ks_list, "popid": popid, "Simulator": sim}
        data = pd.DataFrame(dataset)
        data.to_csv('csv/ks_'+d+".csv", index=False)

def __plot_emd():
    cifre=30
    titolo=35
    colors = ['#33BBEE', '#fc6333']
    sns.set_palette(sns.color_palette(colors))
    fig=plt.figure(figsize=(32, 18), dpi=300)
    #left, width = 0.004, .5 
    #bottom, height = .0, .83
    #right = left + width
    #top = bottom + height
    layer=['L2/3E', 'L2/3I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']

    firing_rate = pd.read_csv("csv/emd_firing_rate.csv")
    cv_isi = pd.read_csv("csv/emd_cv_isi.csv")
    correlation = pd.read_csv("csv/emd_correlation.csv")

    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.5)
    #plt.suptitle("EMD comparison", fontsize = titolo + 5)

    ax1 = plt.subplot(gs[0, :2],)
    plt.text(-0.125,1.03,"A", transform=ax1.transAxes, weight="bold", fontsize=titolo)
    plt.title(r'EMD firing rate', fontsize=titolo+3)
    sns.boxplot(x="popid", y="EMD", hue="Simulator", data=firing_rate)

    ax1.set_xlabel("")
    ax1.set_ylabel("EMD [spikes/s]", fontsize=cifre+2)
    ax1.set_xticks(np.arange(len(layer)))
    ax1.set_xticklabels(layer)
    ax1.tick_params(labelsize=cifre)
    plt.grid(ls='--', axis='y')
    plt.legend([],[], frameon=False)

    ax2 = plt.subplot(gs[0, 2:])
    plt.text(-0.125,1.03,"B", transform=ax2.transAxes, weight="bold", fontsize=titolo)
    plt.title(r'EMD CV ISI', fontsize=titolo+3)
    sns.boxplot(x="popid", y="EMD", hue="Simulator", data=cv_isi)
    ax2.set_xlabel("")
    ax2.set_ylabel("EMD", fontsize=cifre+2)
    ax2.set_xticks(np.arange(len(layer)))
    ax2.set_xticklabels(layer)
    plt.grid(ls='--', axis='y')
    ax2.tick_params(labelsize=cifre)
    plt.legend([],[], frameon=False)

    ax3 = plt.subplot(gs[1, 1:3])
    plt.text(-0.125,1.03,"C", transform=ax3.transAxes, weight="bold", fontsize=titolo)
    plt.title(r'EMD correlation', fontsize=titolo+3)
    sns.boxplot(x="popid", y="EMD", hue="Simulator", data=correlation)

    ax3.set_xlabel("")
    ax3.set_ylabel("EMD", fontsize=cifre+2)
    ax3.set_xticks(np.arange(len(layer)))
    ax3.set_xticklabels(layer)
    plt.grid(ls='--', axis='y')
    ax3.tick_params(labelsize=cifre)
    plt.legend(bbox_to_anchor=(1.6, 0.5),loc='center right', prop={'size': titolo})
    plt.subplots_adjust(top=0.9, hspace = 0.25)
    
    plt.savefig("emd_boxplot.pdf")

def __plot_ks():
    cifre=30
    titolo=35
    colors = ['#33BBEE', '#fc6333']
    sns.set_palette(sns.color_palette(colors))
    fig=plt.figure(figsize=(32, 18), dpi=300)
    #left, width = 0.004, .5 
    #bottom, height = .0, .83
    #right = left + width
    #top = bottom + height
    layer=['L2/3E', 'L2/3I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']

    firing_rate = pd.read_csv("csv/ks_firing_rate.csv")
    cv_isi = pd.read_csv("csv/ks_cv_isi.csv")
    correlation = pd.read_csv("csv/ks_correlation.csv")

    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.5)

    ax1 = plt.subplot(gs[0, :2],)
    plt.text(-0.125,1.03,"A", transform=ax1.transAxes, weight="bold", fontsize=titolo)
    plt.title(r'KS test firing rate', fontsize=titolo+3)
    sns.boxplot(x="popid", y="KS", hue="Simulator", data=firing_rate)

    ax1.set_xlabel("")
    ax1.set_ylabel("KS test", fontsize=cifre+2)
    ax1.set_xticks(np.arange(len(layer)))
    ax1.set_xticklabels(layer)
    ax1.tick_params(labelsize=cifre)
    plt.grid(ls='--', axis='y')
    plt.legend([],[], frameon=False)

    ax2 = plt.subplot(gs[0, 2:])
    plt.text(-0.125,1.03,"B", transform=ax2.transAxes, weight="bold", fontsize=titolo)
    plt.title(r'KS test CV ISI', fontsize=titolo+3)
    sns.boxplot(x="popid", y="KS", hue="Simulator", data=cv_isi)
    ax2.set_xlabel("")
    ax2.set_ylabel("KS test", fontsize=cifre+2)
    ax2.set_xticks(np.arange(len(layer)))
    ax2.set_xticklabels(layer)
    plt.grid(ls='--', axis='y')
    ax2.tick_params(labelsize=cifre)
    plt.legend([],[], frameon=False)

    ax3 = plt.subplot(gs[1, 1:3])
    plt.text(-0.125,1.03,"C", transform=ax3.transAxes, weight="bold", fontsize=titolo)
    plt.title(r'KS test correlation', fontsize=titolo+3)
    sns.boxplot(x="popid", y="KS", hue="Simulator", data=correlation)

    ax3.set_xlabel("")
    ax3.set_ylabel("KS test", fontsize=cifre+2)
    ax3.set_xticks(np.arange(len(layer)))
    ax3.set_xticklabels(layer)
    plt.grid(ls='--', axis='y')
    ax3.tick_params(labelsize=cifre)
    plt.legend(bbox_to_anchor=(1.6, 0.5),loc='center right', prop={'size': titolo})
    plt.subplots_adjust(top=0.9, hspace = 0.25)
    
    plt.savefig("ks_boxplot.pdf")





