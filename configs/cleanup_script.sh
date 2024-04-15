#!/bin/bash -l

read -p "Are you sure? " -n 1 -r 
echo #(optional) move to a new line

if [[ $REPLY =~ ^[Yy]$ ]]
then
    # do dangerous stuff

rm *.err
rm *.out
rm *.log
rm *.sdf
rm *.csv
rm -r *compounds
rm -r *Compounds
rm -r *CheckPoints
rm JSON
rm docking_counter.txt
rm *.check

rm test_ligand.sq
rm -r reinvent_results
rm -r ROCS_JSON
rm -r ROCS_CSV
rm -r ROCS_SDF
rm -r saved_states
rm -r reinvent_logging
rm -r docked_files
rm -r output
rm -r reinventBatch
rm -r surrogateCheckpoints
rm -r virtualLibrary 

fi
