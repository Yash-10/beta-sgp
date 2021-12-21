################################################################################
#								   	       									   #
#									       									   #
#	Removes bad biases and bad flat images. Keeps good images intact           #
#									       									   #
#	Bad bias removal: STDDEV > 1.5 * (median of STDDEV of all bias)	           #
#	Bad flats removal: MEAN > 50,000				       					   #
#									       									   #
################################################################################

# Description of the Automation.cl code to remove bad biases and flats
#Algorithm used:
#``````````````
#1) To remove bad bias images:
#Remove images with STDDEV &gt; 1.5 * (Median of STDDEV of all biases)
#2) To remove bad flat images:
#Remove images with MEAN &gt; 50,000
#** Removes any lists already present to prevent overwriting or appending to existing files.
#** Selects bias images based on the &quot;exptime&quot; keyword which is 0 by default - Ensures reliability
#of selection. Selecting based on &quot;object?=&#39;bias&#39;&quot; may not be the best choice as the &#39;object&#39;
#keyword may not be uniformly fixed.
#** Fixes the FILTER keyword of bias images to &#39;1 FREE&#39;, if required.
#** Stores bad biases and flats inside the folder &#39;BadFrames&#39;.


# Remove any lists/directories already present to prevent overwriting/appending to existing files/directories
if (access("list")) {
	!rm -f list
}
if (access("biases")) {
	!rm -f biases
}
if (access("flats")) {
	!rm -f flats
}
if (access("object")) {
	!rm -f object
}
if (access("stddev_list")) {
	!rm -f stddev_list
}
if (access("$BadFrames")) {
	!rm -r BadFrames
}
;

print("\nMaking lists...")
print("---------------")

# Create list of all .fits images
ls *.fits > list


# Bias -list
hselect @list $I "exptime?=0.0" > biases

# Flat -list
hselect @list $I "object?='flat'" > flats

# Object -list
hselect @list $I "object?='M13'" > object

print("\n\tList making complete!")

# Initialize sum of stddev for all bias images to zero
real all_stddev = 0
real std_dev = 0

struct *b_list = "biases"
struct image

print("\nChecking filter keywords for Bias images")
print("---------------")

while (fscan(b_list, image) != EOF) {
	imgets (image, "filter")
	if (imgets.value != "1 Free") {
		# Sets filter value to 1 Free for all biases		
		hedit (image, "filter", "1 Free", add+, ver-)
	}
	imstat (image, field="stddev", format-) | scan(std_dev)
	all_stddev  = all_stddev + std_dev
}
print("\n\tBias filter keywords fixed!")

!rm -f biases
hselect @list $I "exptime?=0.0" > biases
struct images
struct *bias_list = "biases"

# Store std dev in a separate file
while (fscan(bias_list, images) != EOF) {
	imstat (images, field="stddev", format-, >> "stddev_list")
}

list = "stddev_list"
real stddev_
real median_stddev

int N = 0 # No. of entries in the file "stddev_list" 
int N2 = 0 # Just defined a new var
print ("!more biases | wc -l") | cl() | scan(j)
int nbias
nbias = j

real stddev1
real stddev2

# Calculate median standard deviation

if (nbias % 2 == 0) {
	while (fscan(list, stddev_) != EOF) {
		N += 1
		if (N == nbias / 2) {
			stddev1 = stddev_
		}
		else if (N == (nbias / 2) + 1) {
			stddev2 = stddev_
		}
	}
median_stddev = (stddev1 + stddev2) / 2
}

else {
	while (fscan(list, stddev_) != EOF) {
		N2 += 1
		if (N2 == (nbias + 1) / 2) {
			median_stddev = stddev_
		}
		;
	}
}

# Outlier Detection

print("\nAnalyzing Bias images...")
print("---------------")

# Create a directory to store bad images
if (!access("$BadFrames")) {
	!mkdir BadFrames
}
;

struct image1
real std__dev
struct *b__list = "biases"

print("!more biases | wc -l") | cl() | scan(q)
print("\n\tInitial number of biases: ", q)

# Remove bad biases 

while (fscan(b__list, image1) != EOF) {
	imstat (image1, field="stddev", format-) | scan(std__dev)
	if (std__dev >  1.5 * median_stddev && (!access("$BadFrames/" + image1))) {
		movefiles (image1, "BadFrames")
	}
	;
}

# After removing bad biases, make list of remaining images
!rm -f list
!rm -f biases
ls *.fits > list
hselect @list $I "exptime?=0.0" > biases

print("\n\tBias images analyzed!")

print("!more biases | wc -l") | cl() | scan(k)
int nbiases
nbiases = k
print("\n\tFinal number of biases: ", k)

print("\nAnalyzing Flat images...")
print("---------------")

struct *f_list = "flats"
struct imagef
real mean_

print("!more flats | wc -l") | cl() | scan(r)
print("\n\tInitial number of flats: ", r)

# Remove bad flats

while (fscan(f_list, imagef) != EOF) {
	imstat (imagef, field="mean", format-) | scan(mean_)
	if (mean_ > 50000) {
		movefiles (imagef, "BadFrames")
	}
	;
}

# After removing bad flats, make list of remaining images
!rm -f list
!rm -f flats
ls *.fits > list
hselect @list $I "object?='flat'" > flats

print("\n\tFlat images analyzed!")

int fcount
print("!more flats | wc -l") | cl() | scan(fcount)
print("\n\tFinal number of flats: ", fcount)

print("\nProcess completed!")
