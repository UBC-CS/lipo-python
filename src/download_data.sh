#!/bin/bash
#
# script to grab files line by line 
# from the urls in provided a txt file
#
# Usage: 
#   - bash download_data.sh urls.txt data

while read line; 
do 
location="$2/$(basename $line)"
wget $line -O $location
done <$1