# BMN Project

Current status
- code seems to be working, BUT
    - only converges stochastically
    - seems to converge less often when L increases
    - I don't understand the commutator expression for [H,O] constraints
- BMN might need complex vectors


# TODO
- coding
    - clean up
        - remove unnecessary comments, debugging options
        - docstrings, typing
    - make implementation more efficient
    - how to set-up scans, logging
- physics
    [x] verify 1-matrix model
    - maybe verify 2-matrix model
    - mini-BMN model
        - what does a giant graviton look like?
        - what role does SUSY play here, even with no fermions?
        - what, if any, regimes or coupling configurations do we have analytic tractability?
    - how can I handle finite-temperature
        - I think the constraints are the same, but check this
        - what would I optimize for?
    - Maldacena-Milekhin conjecture

## LOG

### June 10
- Constraints are not symmetric - [H, O] results in a different expression than [O, H]
    - I think I understand this, and it's expected
    - I implemented my own function to compute [H, 0], and it agrees with Han's for one ordering choice and the 1-matrix case
        - does not pass the unit test I wrote that utilized the two-matrix example
    - I suspect that the commutator relation I worked out is valid for abstract operators, but that it does not respect the ordering of the gauge indices imposed by the single-trace property. Look into this more.
- I am not matching Han et al in terms of constraints
- Start playing with the two-matrix problem and add SO(2) invariance
- For mini-bmn, worry about the imaginary term in the Hamiltonian.

## Pseudocode of SDP optimization
first do sdp_init
    Finds the parameters such that
    1. All bootstrap tables are positive semidefinite;
    2. ||A.dot(param) - b||^2_2 + reg * ||param - init||^2_2 is minimized.

then for loop for iter in range(max_iters)

    sdp_relax
        Finds the parameters such that
        1. All bootstrap tables are positive semidefinite;
        2. ||param - init||_2 <= 0.8 * radius;
        3. The violation of linear constraints ||A.dot(param) - b||_2 is minimized.

    compute value and gradient of quadratic constraints

    sdp_minimize
        Finds the parameters such that
        1. All bootstrap tables are positive semidefinite;
        2. ||param - init||_2 <= radius;
        3. A.dot(param) = b;
        4. vec.dot(param) + reg * np.linalg.norm(param) is minimized.

    possibly shrink radius

    compute smallest eigenvalue

    consider adjusting mu

    accept or reject update