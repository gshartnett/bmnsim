# BMN Project

## Installation Note
To install sparseqr on my M2 Macbook Air:
- I manually copied the files in the `sparseqr` directory of the https://github.com/yig/PySPQR repo
- I then did `import sparseqr` within Python
- this failed due to libraries not being installed. Following this SE post: https://stackoverflow.com/questions/73479899/sklearn-cant-find-lapack-in-new-conda-environment, I then did
    - `conda install openblas`
    - `brew install lapack`
    - `pip uninstall numpy`
    - `pip install numpy==1.21.1`
    - `conda update --all`