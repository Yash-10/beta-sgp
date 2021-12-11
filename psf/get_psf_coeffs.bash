#!/bin/bash 

# Given a list of PSF binary coefficients file, this program writes the output to a .txt file.
# The "tmpsf" folder needs to placed in the directory from where the script is run.


##### User inputs #####
bin_list="psf_bin_list"
#######################

for im in `cat ${bin_list}`;
do	
	actual_name="${im}"
	name="${im}.txt"
	tmpsf/tmpsf ${actual_name} ${name}
done
echo "DONE!"
exit 0
