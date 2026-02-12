#!/bin/bash
# shopt -s 

# Step 1: Run first version
echo "Running low_dl_fr_0..."
# conpy10
py sohini_test.py > paper_plots/low_dl_fr_0/log.txt &

# Store the process ID of the background job
PID1=$!

# Wait for the first process to finish
wait $PID1

# Step 2: Switch to second version
sed -i '' 's/^# \(.*low_dl_low_fr.*# low_dl_low_fr\)/\1/' sohini_test.py
sed -i '' 's/^# \(.*(6,3).*#low dl low fr\)/\1/' sohini_test.py
sed -i '' 's/^\(.*low_dl_fr_0.*# low_dl_fr_0\)/# \1/' sohini_test.py
sed -i '' 's/^\(.*(6,0).*#low dl fr 0\)/# \1/' sohini_test.py

PID2=$!

# Wait for the second process (optional)
wait $PID2

# Step 4: Run second version
echo ""
echo "Running low_dl_low_fr..."
py sohini_test.py > paper_plots/low_dl_low_fr/log.txt &
