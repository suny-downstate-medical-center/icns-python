# Intrinsic Cardiac Nervous System - Principal Neuron model - Python
Version 0.1

The mod files are from here:
https://senselab.med.yale.edu/ModelDB/ShowModel?model=3800

# Contents
PN_model.py; README.md, example_plot_python.png

# Usage
## Install NEURON
Enter:
pip3 install neuron

## Compile the mod files
Enter: 
nrnivmodl mod

## Install NetPyNE
Clone the repo by entering:
git clone https://github.com/Neurosim-lab/netpyne.git

Install the repo, enter:
pip3 install -e netpyne

## Run a simulation
Enter:
python3 -i PN_model.py

## Output
Generates voltage v. time plot (example_plot_python.png)

