# VenusCoreModelling

This repository contains the raw python code for creating internal profiles of Venus's structure, and determining optimal core structure to fit estimate of moment of intertia and mass.

The code uses numpy, scipy, matplotlib, and (importantly) is based off the burnman code (available here: https://github.com/geodynamics/burnman).

The file VenusStructureMonteCarlo.py performs the optimisation, producing a datafile of Inner/Outer core optimised radii for a range of input variables/compositions.

VenusPlotStructure.py creates a plot of (two) internal structures, and calculate internal temperatures from Aitta (2012), and plots against 2 solidii (Steinbrugge et al. 2021 and Anzellini et al., 2013).

The reference file other.py is a modification of the Burnman mineral physics database file to incorporate new Vinet-style EOS for FeSi (solid: Brosh et al. 2009; liquid: Steinbrugge et al. 2020).




