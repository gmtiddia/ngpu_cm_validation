# NEST GPU Validation for Cortical Microcircuit model simulations

Here is presented a fast way for compare the results in terms of spiking activity between NEST GPU and NEST for the simulation of the Cortical Microcircuit model (Potjans, 2014). 

The validation takes in input the files of the spike times for the simulations and returns the distributions of firing rate, CV ISI and Pearson correlation for every population of the model. The distributions obtained with NEST and NEST GPU are shown side by side using boxplots or violinplots, and are quantitatively compared using the Earth Mover's Distance (EMD) metric.

## What you need to start the validation

To perform the validation you need:
- 10 NEST GPU simulations with different seeds for random number generation
- 20 NEST simulations with different seeds for random number generation, splitted into two sets of 10 simulations each

All the simulations should have enabled the recording of the spikes and should have 500 ms of presimulation and 10000 ms of simulation. The simulations sets should be stored in separate direcotories, and the results of each simulation of the set should be saved into separate folders (data0, data1, ..., data9).

To save the spike times files obtained with NEST in the same way as NEST GPU you can run the bash script ``merge_st.sh`` chosing the local number of threads used to perform NEST simulations. The bash script has to be run in the directory containing all the simulation of the set (i.e. the one containinf data0, data1, ..., data9).

You can find the bash script ``merge_st.sh`` in the directory ``miscellaneous``, together with the bash script ``run_simulations.sh`` which is a sample script that can be used to run the simulations.

## Configuration

To set all the steps of the validation process you can edit the Python script ``val_config.py`` which contains all the information needed to perform the validation. You can edit the number of simulations for each set, the paths for the simulation directories and boolean parameters that can enable or disable the computation of the distribution plots and the EMD boxplots.

## Run validation

Once the NEST spike times are adapted using ``merge_st.sh`` and the configuration file is edited, run the Python script ``run_validation.py`` to perform the validation.

