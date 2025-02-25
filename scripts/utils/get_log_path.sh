#!/bin/bash

# Get current date and time
datetime=$(date '+%Y%m%d%H%M%S')

# Extract components
year=${datetime:0:4}
month=${datetime:4:2}
day=${datetime:6:2}
hour=${datetime:8:2}
minute=${datetime:10:2}
second=${datetime:12:2}

# Construct the path
path="${year}.${month}.${day}/${hour}.${minute}.${second}"

# Print the path
echo "$path"