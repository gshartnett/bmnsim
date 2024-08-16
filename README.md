# BMN Project

## TODO
- MiniBMN
    - OOM issues with L=3
        - can I improve this or resort to dropping some operators
- Physics/Understanding
    - giant gravitons
    - relate BFSS data to published papers
        - how to compare my data to finite N, finite L lattice results?
        - Maldacena-Milekhin conjecture
    - consider reaching out to Jorge and Oscar
- Fermions
    - understand general approach
    - what woudld this enable? SYK?
- Coding
    - clean up
        - remove unnecessary comments, debugging options
        - docstrings, typing
    - make implementation more efficient
    - how to set-up scans, logging
    - revisit how I'm computing [H,O] commutators, make sue I understand
    - config handling
        - revisit directory naming of checkpoints vs configs, vs data
    - model class
        - do I need to have all the hard-coded model classes? Should I just have
            - 1 complex class for BMN
            - 1 real for 1-, 2-, 3-, and BFSS models?

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