# Pre-CAT
The purpose of this code is to reconstruct and analyze preclinical cardiac CEST data acquired using the method described in the manuscript "Ungated, plug-and-play preclinical cardiac CEST-MRI using radial FLASH with segmented saturation"

![Pixelwise preview](/instructions/img/maps.png)

## Requirements

### Dependencies 
* Scripts tested on Python 3.9.16
* Dependencies listed in requirements.txt
* A Berkeley Advanced Reconstruction Toolbox (BART) installation is required for reconstruction
  * Install the latest version of BART by following the instructions listed at: https://mrirecon.github.io/bart/installation.html
    * For Mac M1 installation instructions, refer to: https://github.com/mrirecon/bart/issues/326
  * Make sure that BART_TOOLBOX_PATH is set and added to the PATH variable

  #### Suggested installation routes
  * [pip](https://pip.pypa.io/en/stable/)
    * Use ```pip install -r /path/to/requirements.txt```
  * [Anaconda](https://www.anaconda.com/products/distribution)
    * Use ```conda install --file /path/to/requirements.txt``` 


## Instructions
Detailed instructions can be found in the instructions folder.

![Segmentation preview](/instructions/img/roi.png)

#### Note: Pixelwise mapping is not included in main script. Refer to /scripts/plot_maps.py for pixelwise mapping.

## References
Martin Uecker, Frank Ong, Jonathan I Tamir, Dara Bahri, Patrick Virtue, Joseph Y Cheng, Tao Zhang, and Michael Lustig. Berkeley Advanced Reconstruction Toolbox. Annual Meeting ISMRM, Toronto 2015, In Proc. Intl. Soc. Mag. Reson. Med. 23:2486 

### This repository is associate with the following paper, please cite this work when using the code published in this repository:

Weigand-Whittier J, Wendland M, Lam B, et al. Ungated, plug-and-play cardiac CEST-MRI using radial FLASH with segmented saturation. Magn Reson Med. 2024. 10.1002/mrm.30382


