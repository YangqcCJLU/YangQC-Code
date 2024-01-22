Detection of Drug Residues in Bean Sprouts by Hyperspectral Imaging Combined with 1DCNN with Channel Attention Mechanism  
The 'code' folder contains code written in MATLAB R2021b and python 3.9:  
	'matlab' folder:  
 		Functions that may be used during data processing (functions for data processing);
   		Hyperspectral Data Processing Toolbox (HSI_toolbox);
     		Processing flow of two machine learning models (machine_learning.m).
	'python3' folder：
 		Processing flow for deep learning models (deep_learning.py);
   		Data processing functions that may be used (func_pre.py);
     		A folder of several pieces of code for data visualization (visualization).

*Note: The files read in the code can be found in the 'data.zip' folder with the corresponding file names.

'dataset' folder：
	'HSI data (5 classes)' folder：Black and white corrected hyperspectral data for five types of samples ('.hdr' format )；
	'datasets for model training&test' folder：MATLAB Output for Model Training and Testing ('.mat' format)；Extracted ROI data（'.xlsx' format）；
	'results of feature extraction' folder：Spectral features extracted in MATLAB；
	'spectra' folder：Contains raw spectra, mean spectra, standard deviation of spectra for each group of samples, band information；
	'confusion matrix' folder：Confusion matrix for four models
