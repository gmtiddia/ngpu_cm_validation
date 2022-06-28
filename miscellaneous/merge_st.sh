read -p "Enter local number of threads used in NEST: " threads
threads=$((threads - 1))

for i in $(seq 0 9); do
	# loop over the populations of the model
    for j in $(seq 0 7); do
	echo 'sender time_ms' > data$i/spike_times_$j.dat
	# loop over the local number of threads (here 80)
	for th in $(seq -w 0 $threads); do
	    tail -n +4 data$i/spike_recorder-7717${j}-$th.dat >> data$i/spike_times_$j.dat
	done
    done
done
