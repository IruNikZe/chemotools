from math import factorial
from numbers import Integral

import numpy as np
from sklearn.utils import check_scalar


def _calc_arbitrary_fin_diff_kernel(
    *,
    grid_points: np.ndarray,
    differences: int,
) -> np.ndarray:
    # the number of grid points is counted
    num_grid_points = grid_points.size

    # if the grid points cannot support the respective difference, an error is raised
    if differences >= num_grid_points:
        raise ValueError(
            f"\n{num_grid_points} grid points cannot support a {differences}-th order "
            f"difference."
        )
    # else nothing

    # then, the system of linear equations to solve is set up as A@x = b where x is
    # the kernel vector
    lhs_mat_a = np.vander(x=grid_points, N=num_grid_points, increasing=True).T
    rhs_vect_b = np.zeros(shape=(num_grid_points,), dtype=np.float64)
    rhs_vect_b[differences] = factorial(differences)

    # the kernel is computed and returned
    return np.linalg.solve(a=lhs_mat_a, b=rhs_vect_b)


def calc_forward_diff_kernel(
    *,
    differences: int,
    accuracy: int = 1,
) -> np.ndarray:
    """Computes the kernel for forward finite differences which can be applied to a
    series by means of a convolution, e.g.,

    ```python
        kernel, _, _ = calc_forward_fin_diff_kernel(differences=2, accuracy=1)
        differences = np.convolve(series, kernel) # boundaries require special care
    ```

    Parameters
    ----------
    differences : int
        The order of the differences starting from 0 for the original curve, 1 for the
        first order, 2 for the second order, ..., and ``m`` for the ``m``-th order
        differences.
        Values below 0 are not allowed.

    accuracy : int, default=1
        The accuracy of the approximation which must be a positive integer starting
        from 1.

    Returns
    -------
    fin_diff_kernel : np.ndarray
        A NumPy-1D-vector resembling the kernel from the code example above.

    Raises
    ------
    ValueError
        If the difference order is below 0, the accuracy is below 1, or the number of
        grid points is not sufficient to support the respective difference order.

    """
    # the input is validated
    check_scalar(
        differences,
        name="differences",
        target_type=Integral,
        min_val=0,
        include_boundaries="left",
    )
    check_scalar(
        accuracy,
        name="accuracy",
        target_type=Integral,
        min_val=1,
        include_boundaries="left",
    )

    # afterwards, the number of grid points is evaluated, which is simply the sum of the
    # difference order and the accuracy
    num_grid_points = differences + accuracy

    # then, the system of linear equations is solved for the x in A@x = b since x is
    # the kernel vector
    grid_points = np.arange(
        start=0,
        stop=num_grid_points,
        step=1,
        dtype=np.float64,
    )
    fin_diff_kernel = _calc_arbitrary_fin_diff_kernel(
        grid_points=grid_points, differences=differences
    )

    return fin_diff_kernel


for iter_i in range(1, 7):
    print(calc_forward_diff_kernel(differences=4, accuracy=iter_i))
