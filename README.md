# Specialization project
Branch for the autumn project. Note that this branch is only used as an archive, the updated preprocessing exists in the main branch.

Written by *Jun Xing Li*

## Table of Contents
* [To be updated]()
* [Specialization project](#specialization-project)

## Specialization project ([TKP4580 - Chemical Engineering, Specialization Project](https://www.ntnu.edu/studies/courses/TKP4580/2023#tab=omEmnet))
The specialization/autumn project only takes account for the data analysis and some data 
cleaning. Most of the relevant code is in the notebook and here are some 
explainations. 

1. `initial_cleaning.ipynb`: The first import of the raw dataset and some
cleaning steps are performed, such as removal of some selected columns and rows. The resulting initial cleaned data is saved as a pickle.

2. `finding_stability.ipynb`: Using the cleaned dataset in order to find periods of stability and saves a Excel sheet of the mean values for each case.

3. `generate_plots_spec.ipynb`: Used for generating plots for the final report. Includes a general overview, temeprature profile and parity plot of density.

4. `exploring_data.ipynb`: This notebook contains initial data exploration, including various plotting techniques and a trial of outlier detection using DBSCAN. It is retained for historical reference but is not actively used in the current analysis.
