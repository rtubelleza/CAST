# Summary
This a set of Python packages (.py files) useful for the processing, analysis and simulation of datasets from the neuPrint ecosystem (https://neuprint.janelia.org/). This was mainly for a research project, where I explored using the connectome dataset for retrieving premotor and motor neuron clusters, as well as simulated the entire VNC (MANC) as a virtual nervous system, generating in-silico data. The script used for simulations under many parameters and settings are found in param_search.py.

Documentation for classes within the packages are a current work in progress.

Example usage for how to process the data using the ConnDF class witinh the data_analysis.py package are provided in cast_example.ipynb. More updates to this notebook will be added, such as how to use the analysis packages, run neuron clustering and run the interative UI within this class. 

# Dependencies
As these are mostly wrapper classes, they rely on various Python packages. It is therefore highly recommended to construct a conda environment dedicated for using CAST. An environment.yml file containing a sample environment with the necessary packages is provided.

