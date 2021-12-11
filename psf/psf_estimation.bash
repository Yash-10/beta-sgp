#!/bin/bash

# A script to calculate PSF binary files for each image from a list of files.
# This script is a modified version of the "getpsf" routines from the original DIAPL software.

#--------------- PARAMETERS TO EDIT ------------------
ERR_PFX="\n\tERROR! psf_estm.bash: "
INSTRUMENT="instrument.par"
FITS=`awk '($1=="FITS") {print $3}' ${INSTRUMENT}`
MASK="m"
BIN="/home/yash/DIAPL/DIAPL_BIN"
IMAGES="Iimages.list"
NX=`awk '($1=="NX") {print $3}' ${INSTRUMENT}`
NY=`awk '($1=="NY") {print $3}' ${INSTRUMENT}`
MARG=`awk '($1=="MARG") {print $3}' ${INSTRUMENT}`
#-----------------------------------------------------

echo
echo "Calculating PSF for all images"
echo "------------------------------"
echo ""

for im in `cat ${IMAGES}`;
do
	name=${im%.${FITS}}

	coord_file="${name}.coo"
	psf_file="psf${name}.bin"

	if [ -f "$coord_file" ] ; then
    		rm "$coord_file"
		echo "removed existing file '${coord_file}'"
	fi

	if [ -f "$psf_file" ] ; then
                rm "$psf_file"
                echo "removed existing file '${psf_file}'"
        fi

	if [ -f "${name}c.${FITS}" ]; then
		rm "${name}c.${FITS}"
		echo "removed existing intermediate cutted images"
	fi

	if [ -f "${name}mc.${FITS}" ]; then
		rm "${name}mc.${FITS}"
		echo "removed existing intermediate cutted mask images"
	fi

	echo "${im}"
	echo "--------------------"
	
	(( xl=1+MARG ))
	(( xu=NX+MARG ))
        (( yl=1+MARG ))
        (( yu=NY+MARG ))

	fim=${name}c.${FITS}
	cim=${name}c.coo
	imask=${name}${MASK}.${FITS}
        cmask=${name}${MASK}c.${FITS}
        
	echo cutfitsim ${im} ${fim} ${xl} ${xu} ${yl} ${yu}
        ${BIN}/cutfitsim ${im} ${fim} ${xl} ${xu} ${yl} ${yu}
    	if [ $? -ne 0 ]
    	then
      	  echo -e ${ERR_PFX}"cutfitsim failed - aborting now\n"
      	  exit 24
    	fi

    	echo cutfitsim ${imask} ${cmask} ${xl} ${xu} ${yl} ${yu}
    	${BIN}/cutfitsim ${imask} ${cmask} ${xl} ${xu} ${yl} ${yu}
    	if [ $? -ne 0 ]
    	then
      	  echo -e ${ERR_PFX}"cutfitsim failed - aborting now\n"
          exit 25
    	fi
	
	echo fwhmm fwhmm.par ${fim}
	arg=(`${BIN}/fwhmm fwhmm.par ${fim}`);
    	echo ${arg[*]}

    	sky=${arg[2]}
    	fwhm=${arg[3]}
	
	echo "sfind"
	echo "-----"

	echo sfind sfind.par ${INSTRUMENT} ${fim} ${cmask} ${sky} ${fwhm} ${cim}
        ${BIN}/sfind sfind.par ${INSTRUMENT} ${fim} ${cmask} ${sky} ${fwhm} ${cim}
        if [ $? -ne 0 ]
        then
          echo -e ${ERR_PFX}"sfind failed - aborting now\n"
          exit 13
        fi

	echo "CALCULATE THE PSF COEFFICIENTS ON A SINGLE WHOLE FRAME"
	echo ""

	echo "getpsf"
	echo "------"
	
	echo -n getpsf getpsf.par ${INSTRUMENT} ${fim} ${cmask} ${cim}
    	echo " "psf_${name}.bin
    	${BIN}/getpsf getpsf.par ${INSTRUMENT} ${fim} ${cmask} ${cim} psf_${name}.bin
	
	if [ $? -ne 0 ]; then continue; fi  # Go to next image if PSF calculation fails for current image.
	
	echo "removing unecessary files"
	echo "-------------------------"
        rm -f ${cim}
	rm -f ${fim}
	rm -f ${cmask}

	echo ""	
done

echo "DONE!"
exit 0
