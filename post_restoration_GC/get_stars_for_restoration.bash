#!/bin/bash


#--------------- PARAMETERS TO EDIT ------------------
ERR_PFX="\n\tERROR! get_stars_for_restoration.bash: "
INSTRUMENT="instrument.par"
FITS=`awk '($1=="FITS") {print $3}' ${INSTRUMENT}`
MASK="m"
BIN="/home/yash/DIAPL/DIAPL_BIN"
IMAGES="Iimages.list"
NX=`awk '($1=="NX") {print $3}' ${INSTRUMENT}`
NY=`awk '($1=="NY") {print $3}' ${INSTRUMENT}`
MARG=`awk '($1=="MARG") {print $3}' ${INSTRUMENT}`
#-----------------------------------------------------

CORR="corr.txt"

if ! [ -r diapl_setup.bash ]
then
  echo -e ${ERR_PFX}" cannot read 'diapl_setup.bash'\n"
  exit 1
fi
source diapl_setup.bash
REFIM=${REFIM%.${FITS}}

(( nxw=NX+2*MARG ))
(( nyw=NY+2*MARG ))

(( xload=(NX0+NX-1)/NX ))
(( yload=(NY0+NY-1)/NY ))


echo
echo "Detecting stars on all images"
echo "------------------------------"
echo ""

echo ${RIMAGES}

for iy in `seq 1 ${yload}`
	do
		for ix in `seq 1 ${xload}`
		do
			prefix="_${ix}_${iy}"
			suffix="r"${prefix}.${FITS}
			rimages=${RIMAGES}${prefix}".txt"
			mask=${REFIM}${MASK}${suffix}

			if [ -e ${rimages} ]; then
			if [ ${RMTMP} == "yes" ]; then rm -f ${VERBOSE} ${rimages};
			else mv -f ${VERBOSE} ${rimages} ${rimages}.bak; fi
			fi
			touch ${rimages}

			(( xl=1+NX*(ix-1)-MARG ))
			(( yl=1+NY*(iy-1)-MARG ))
			(( xu=NX*ix+MARG ))
			(( yu=NY*iy+MARG ))

			echo
			echo "--------------------------"
			echo "Section: "${prefix}" [ "${xl}" : "${xu}" , "${yl}" : "${yu}" ]"
			echo "--------------------------"

			echo "Reference image:"
			echo "--------------------------"

			arg=(`grep -F ${REFIM} ${SHIFTS}`)
			if [ $? -ne 0 ]
			then
			echo -e ${ERR_PFX}" grep "${REFIM} ${SHIFTS}" failed - abort\n"
			exit 2
			fi

			sky=${arg[4]}
			fwhm=${arg[5]}

			fim=${REFIM}${prefix}.${FITS}
			imcoo=${REFIM}${prefix}.coo

			(( n11=xl-arg[1] ))
			(( n12=xu-arg[1] ))
			(( n21=yl-arg[2] ))
			(( n22=yu-arg[2] ))

			echo   cutfitsim ${REFIM}.${FITS} ${fim} ${n11} ${n12} ${n21} ${n22}
			${BIN}/cutfitsim ${REFIM}.${FITS} ${fim} ${n11} ${n12} ${n21} ${n22}
			if [ $? -ne 0 ]
			then
			echo -e ${ERR_PFX}" cutfitsim failed - cannot continue\n"
			exit 3
			fi

			echo   cutfitsim ${REFIM}m.${FITS} ${mask} ${n11} ${n12} ${n21} ${n22}
			${BIN}/cutfitsim ${REFIM}m.${FITS} ${mask} ${n11} ${n12} ${n21} ${n22}
			if [ $? -ne 0 ]
			then
			echo -e ${ERR_PFX}" cutfitsim failed - cannot continue\n"
			exit 4
			fi

			echo   sfind sfind.par ${INSTRUMENT} ${fim} ${mask} ${sky} ${fwhm} ${imcoo}
			echo -n "  "
			${BIN}/sfind sfind.par ${INSTRUMENT} ${fim} ${mask} ${sky} ${fwhm} ${imcoo}
			if [ $? -ne 0 ]
			then
			echo -e ${ERR_PFX}" sfind failed - cannot continue\n"
			exit 5
			fi

			echo   im2float ${fim} ${REFIM}${suffix}
			${BIN}/im2float ${fim} ${REFIM}${suffix}
			if [ $? -ne 0 ]
			then
			echo -e ${ERR_PFX}" im2float failed - cannot continue\n"
			exit 6
			fi

			refcoo=${REFIM}${prefix}.coo

			if [ ${RMTMP} == "yes" ]; then rm -f ${VERBOSE} ${fim}; fi

			echo
			echo "All other images"
			echo "--------------------------"

			for name in `awk '{print $1}' ${SHIFTS}`
				do
				arg=(`grep -F ${name} ${SHIFTS}`)
				if [ $? -ne 0 ]
				then
					echo -e ${ERR_PFX}" grep "${name} ${SHIFTS}" failed - next image\n"
					continue
				fi

				im=${arg[0]%.${FITS}}
				echo ${im}
				echo "--------------------------"

				if [ ${im} == ${REFIM} ]
				then
					echo ${REFIM}${suffix} >> ${rimages}
				else
					(( n11=xl-arg[1] ))
					(( n12=xu-arg[1] ))
					(( n21=yl-arg[2] ))
					(( n22=yu-arg[2] ))

					sky=${arg[4]}
					fwhm=${arg[5]}

					fim=${im}${prefix}.${FITS}
					rim=${im}${suffix}
					imcoo=${im}${prefix}.coo
					mim=${im}${prefix}.match
					gim=${im}${prefix}.coeff
					imask=${im}${MASK}.${FITS}
					cmask=${im}${MASK}${prefix}.${FITS}
					rmask=${im}${MASK}${suffix}

					echo   cutfitsim ${arg[0]} ${fim} ${n11} ${n12} ${n21} ${n22}
					${BIN}/cutfitsim ${arg[0]} ${fim} ${n11} ${n12} ${n21} ${n22}
					if [ $? -ne 0 ]
					then
					echo -e ${ERR_PFX}" cutfitsim failed - next image\n"
					continue
					fi

					echo   cutfitsim ${imask} ${cmask} ${n11} ${n12} ${n21} ${n22}
					${BIN}/cutfitsim ${imask} ${cmask} ${n11} ${n12} ${n21} ${n22}
					if [ $? -ne 0 ]
					then
					echo -e ${ERR_PFX}" cutfitsim "${imask}" failed - next image\n"
					exit 7
					fi

					# Note the same fwhm and background values can be used on the image after cutting
					# since the stars themselves are kept intact, only cutting is done.
					echo   sfind sfind.par ${INSTRUMENT} ${fim} ${cmask} ${sky} ${fwhm}
					echo " "${imcoo}
					echo -n "  "
					${BIN}/sfind sfind.par ${INSTRUMENT} ${fim} ${cmask} ${sky} ${fwhm} ${imcoo}
					if [ $? -ne 0 ]
					then
					echo -e ${ERR_PFX}" sfind failed - next image\n"
					continue
					fi

					nstars=(`wc -l ${imcoo}`)
					if (( nstars[0] < MINSTARS ))
					then
					echo -e ${ERR_PFX}" too few stars found - next image\n"
					continue
					fi

					echo   xymatch xymatch.par ${refcoo} ${imcoo} ${mim} ${CORR}
					${BIN}/xymatch xymatch.par ${refcoo} ${imcoo} ${mim} ${CORR}
					if [ $? -ne 0 ]
					then
					echo -e ${ERR_PFX}" xymatch failed - next image\n"
					continue
					fi

					corr=(`tail -1 ${CORR}`)
					echo ${corr[*]}
					rm -f ${VERBOSE} ${CORR}

					if (( corr[1] > EDGE )) || (( corr[2] > EDGE )) || \
					(( corr[1] < -EDGE )) || (( corr[2] < -EDGE ))
					then
					echo -e "\n\tShift greater than the value of EDGE: correct\n"

					(( n11=xl-arg[1]-corr[1] ))
					(( n12=xu-arg[1]-corr[1] ))
					(( n21=yl-arg[2]-corr[2] ))
					(( n22=yu-arg[2]-corr[2] ))

					# echo   cutfitsim ${arg[0]} ${fim} ${n11} ${n12} ${n21} ${n22}
					# ${BIN}/cutfitsim ${arg[0]} ${fim} ${n11} ${n12} ${n21} ${n22}
					# if [ $? -ne 0 ]
					# then
					# 	echo -e ${ERR_PFX}" cutfitsim failed - next image\n"
					# 	continue
					# fi

					# echo   cutfitsim ${imask} ${cmask} ${n11} ${n12} ${n21} ${n22}
					# ${BIN}/cutfitsim ${imask} ${cmask} ${n11} ${n12} ${n21} ${n22}
					# if [ $? -ne 0 ]
					# then
					# 	echo -e ${ERR_PFX}" cutfitsim failed - next image\n"
					# 	continue
					# fi

					# echo   sfind sfind.par ${INSTRUMENT} ${fim} ${cmask} ${sky} ${fwhm}
					# echo " "${imcoo}
					# echo -n "  "
					# ${BIN}/sfind sfind.par ${INSTRUMENT} ${fim} ${cmask} ${sky} ${fwhm} ${imcoo}
					# if [ $? -ne 0 ]
					# then
					# 	echo -e ${ERR_PFX}" sfind failed - next image\n"
					# 	continue
					# fi

					# nstars=(`wc -l ${imcoo}`)
					# if (( nstars[0] < MINSTARS ))
					# then
					# 	echo -e ${ERR_PFX}" too few stars found - next image\n"
					# 	continue
					# fi

					# echo   xymatch xymatch.par ${refcoo} ${imcoo} ${mim} ${CORR}
					# ${BIN}/xymatch xymatch.par ${refcoo} ${imcoo} ${mim} ${CORR}
					# if [ $? -ne 0 ]
					# then
					# 	echo -e ${ERR_PFX}" xymatch failed - next image\n"
					# 	continue
					# fi
					# rm -f ${VERBOSE} ${CORR}
					fi

					echo
					echo   xygrid xygrid.par ${mim} ${gim}
					${BIN}/xygrid xygrid.par ${mim} ${gim}
					if [ $? -ne 0 ]
					then
					echo -e ${ERR_PFX}" xygrid failed - next image\n"
					continue
					fi

					echo   resamplem resamplem.par ${INSTRUMENT} ${gim} ${fim} ${cmask} ${rim} ${rmask}
					${BIN}/resamplem resamplem.par ${INSTRUMENT} ${gim} ${fim} ${cmask} ${rim} ${rmask}
					if [ $? -ne 0 ]
					then
					echo -e ${ERR_PFX}" resamplem failed - next image\n"
					continue
					fi

					###### On the resampled image, run fwhmm and sfind again for use in other programs/analysis. ######
					echo fwhmm fwhmm.par ${rim}
					arg=(`${BIN}/fwhmm fwhmm.par ${rim}`);
					echo ${arg[*]}

					rsky=${arg[2]}
					rfwhm=${arg[3]}

					rimcoo=${rim%.${FITS}}${prefix}.coo

					# # First remove all previous sfind coordinates that are not going to be used.
					# rm -f ${imcoo}

					# Run sfind again on the resampled frame for use in other programs/analysis.
					echo   sfind sfind.par ${INSTRUMENT} ${rim} ${rmask} ${rsky} ${rfwhm}
					echo " "${rimcoo}
					echo -n "  "
					${BIN}/sfind sfind.par ${INSTRUMENT} ${rim} ${rmask} ${rsky} ${rfwhm} ${rimcoo}
					if [ $? -ne 0 ]
					then
					echo -e ${ERR_PFX}" sfind failed - next image\n"
					continue
					fi

					echo
					echo ${rim} >> ${rimages} 
					if [ ${RMTMP} == "yes" ]; then 
					rm -f ${VERBOSE} ${fim} ${cmask} ${imcoo} ${gim} ${mim}; fi
				fi
			done
		done
	done

echo "DONE!"
exit 0
