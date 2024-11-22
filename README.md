# hydra-pspec
Gibbs sampler-based power spectrum estimation code with foreground filtering and in-painting capabilities

Guide to directories:
	1. figures_and_plots: Have some figures I have plotted
	2. notebooks: Contains all the ipynb files (including ones that can plot results and have run diagnostics on various versions of the code)
	3. gauss_1_data: (Simulated) data containing visibilities for a 1 degree gaussian beam
	4. hera_val: Contains files related to hera validation pipeline
	5. test_data: contains data for airy beam case
	6. test_files: Iteration level outputs for running diagnostics (divergence_tests folders are just results to run diagnostics on "divergence(and non-convergence) of sampler" problem)
	7. systematics_as_gain.pdf : pdf doc explaining the treatment we have implemented in this version of the sampler and why we did what we did


Other callouts:
	1. Please use the plot_results.ipynb notebook to plot the delay power spectra results. The .py scripts may throw errors.
	2. Please check the relative paths in the notebooks (you might need to add a '..' to get it to work because I have just moved them to a new directory)
	3. When switching between data sets, make sure the right 'clean visibilities' are used to calculate the systematic model. This is done in line 914 of hydra_pspec/pspec.py . For airy beam, the data should be test_data/vis-eor-fgs.uvh5 and for gauss 1 degree beam it should be gauss_1_data/vis-eor-ptsrc-gsm.uvh5. This needs to be done manually at the moment. 
