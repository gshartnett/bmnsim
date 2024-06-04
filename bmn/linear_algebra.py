from numbers import Number

import numpy as np
from scipy.linalg import null_space
from scipy.sparse import coo_matrix
from scipy.linalg import svd


def create_sparse_matrix_from_dict(
    index_value_dict: dict[tuple[int, int], Number], matrix_shape: tuple[int, int]
) -> coo_matrix:
    """
    Create a sparse COO-formatted matrix from an index-value dictionary.

    Parameters
    ----------
    index_value_dict : dict[tuple[int, int], Number]
        The index-value dictionary, formatted as in:

        index_value_dict = {
            (0, 1): 4,
            (1, 2): 7,
            (2, 0): 5,
            (3, 3): 1
        }

    matrix_shape: tuple[int, int]
        The matrix shape, required because the matrix is sparse.

    Returns
    -------
    coo_matrix
        The sparse COO matrix.
    """

    # prepare the data
    row_indices = []
    column_indices = []
    data_values = []

    for (row, col), value in index_value_dict.items():
        row_indices.append(row)
        column_indices.append(col)
        data_values.append(value)

    # convert lists to numpy arrays
    row_indices = np.array(row_indices)
    column_indices = np.array(column_indices)
    data_values = np.array(data_values)

    # create the sparse matrix
    sparse_matrix = coo_matrix(
        (data_values, (row_indices, column_indices)), shape=matrix_shape
    )

    return sparse_matrix


def get_null_space(matrix: np.matrix, tol: float = 1e-10) -> np.ndarray:
    """
    Get the null space of a rectangular matrix M.
    TODO can I make this work for sparse matrices?

    Parameters
    ----------
    matrix : np.matrix
        The matrix, with shape (m, n)
    tol : float, optional
        A tolerance, by default 1e-10

    Returns
    -------
    np.ndarray
        A matrix of null vectors K, such that
            M . K = 0
        The shape of K is (n, dim(Null(M)))

    Raises
    ------
    ValueError
        _description_
    """
    null_space_matrix = null_space(matrix)
    verification_result = matrix.dot(null_space_matrix)
    if not np.max(np.abs(verification_result)) <= tol:
        raise ValueError("Warning, null space condition not satisfied.")
    return null_space_matrix


def get_row_space(matrix: np.ndarray, tol: float = 1e-10) -> np.ndarray:

    # perform SVD on the matrix
    U, S, Vt = svd(matrix)

    # determine rank of matrix
    rank = np.sum(
        S > tol
    )  # Consider singular values greater than a small threshold as non-zero

    # extract basis of the row space from the top rank rows of Vt
    row_space_basis = Vt[:rank, :]

    return row_space_basis


def is_in_row_space(matrix: np.ndarray, vector: np.ndarray) -> bool:
    if matrix.shape[1] != len(vector):
        raise ValueError("Error, dimension mismatch.")
    row_space_matrix = get_row_space(matrix)

    # Project the vector v onto the row space
    # Solve the linear system row_space_basis.T * c = v to find the coefficients c
    c, residuals, rank, s = np.linalg.lstsq(row_space_matrix.T, vector, rcond=None)

    # Reconstruct v from the row space basis using the coefficients c
    v_projected = row_space_matrix.T @ c

    return np.allclose(v_projected, vector)
