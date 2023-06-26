#!/bin/bash
# May 18 2023
# Alrieza Niksejel (alireza.niksejel@dal.ca)

# provides the required station_list.json file
# station info file needs to follow the below format:
# longitude(deg) latitude(deg) network station channel depth(km)



# Full path to the dataset
data_dir="/media/dalquake/T7/3k_OBS_validation_dataset/Github/dataset"	
# Full path to the station info file		
station_info="/media/dalquake/T7/3k_OBS_validation_dataset/Github/stations.dat"	


pickhome=`pwd`
[[ ! -d $data_dir ]] || [[ ! -f $station_info ]] && echo "requirement(s) missing" && exit
[[ -d json ]] && rm -r json/* 2> /dev/null || mkdir json

cd $data_dir
n=`grep -cE '^[1-9]|^-' $station_info`
i=1
for sta in `find ./ -type f -name "*mseed" | cut -f 3 -d / | cut -f 2 -d . | sort | uniq`; do
	net=`grep "$sta" $station_info | awk '{print $3}' | head -1`
	cha=`grep "$sta" $station_info | awk '{print $5}' | head -1`
	lat=`grep "$sta" $station_info | awk '{print $2}' | head -1`
	lon=`grep "$sta" $station_info | awk '{print $1}' | head -1`
	dep=`grep "$sta" $station_info | awk '{print $6}' | head -1`
	if [[ $i = 1 ]]; then
		echo "{\"$sta\": {\"network\": \"$net\", \"channels\": [\""$cha"1\", \""$cha"2\", \""$cha"Z\"], \"coords\": [$lat, $lon, $dep]}," >> $pickhome/json/station_list.json
	elif [[ $i = $n ]]; then
		echo "\"$sta\": {\"network\": \"$net\", \"channels\": [\""$cha"1\", \""$cha"2\", \""$cha"Z\"], \"coords\": [$lat, $lon, $dep]}}" >> $pickhome/json/station_list.json
	else
		echo "\"$sta\": {\"network\": \"$net\", \"channels\": [\""$cha"1\", \""$cha"2\", \""$cha"Z\"], \"coords\": [$lat, $lon, $dep]}," >> $pickhome/json/station_list.json
	fi
	echo "done with station $sta"
	i=$((i+1))
done
