import numpy as np
from scipy.sparse import eye as speye

from chemotools._utils.banded_linalg import (
    conv_to_lu_banded_storage,
    lu_banded,
    lu_solve_banded,
    slogdet_lu_banded,
)
from chemotools._utils.finite_differences import (
    calc_forward_diff_kernel,
    forward_finite_diff_conv_matrix,
)
from tests.fixtures import reference_finite_differences  # noqa: F401


def test_forward_diff_kernel(
    reference_finite_differences: list[tuple[int, int, np.ndarray]]  # noqa: F811
) -> None:
    # Arrange
    for differences, accuracy, reference in reference_finite_differences:
        # Act
        kernel = calc_forward_diff_kernel(differences=differences, accuracy=accuracy)

        # Assert
        assert kernel.size == reference.size, (
            f"Difference order {differences} with accuracy {accuracy} "
            f"expected kernel size {reference.size} but got {kernel.size}"
        )
        assert np.allclose(kernel, reference, atol=1e-8), (
            f"Difference order {differences} with accuracy {accuracy} "
            f"expected kernel\n{reference.tolist()}\n"
            f"but got\n{kernel.tolist()}"
        )


def test_forward_finite_diff_conv_matrix() -> None:
    """Tests the generated convolution matrix for forward finite differences by
    comparing it to NumPy's ``convolve``.
    """
    # the random number generator is seeded
    np.random.seed(seed=42)

    # the difference orders and accuracies are looped over for signals of different size
    # to compute the convolution by ordinary convolution and by matrix multiplication
    for size in [100, 1000, 10000, 100000]:
        series = np.random.rand(size)

        for diff in range(0, 20):
            for acc in range(1, 20):
                # the kernel is computed ,,,
                kernel = calc_forward_diff_kernel(differences=diff, accuracy=acc)
                # ... and the random series is convolved with the kernel ...
                # NOTE: the kernel is flipped because of the way NumPy's convolve works
                numpy_convolved_series = np.convolve(
                    series, np.flip(kernel), mode="valid"
                )

                # the convolution matrix is computed ...
                conv_matrix = forward_finite_diff_conv_matrix(
                    differences=diff, accuracy=acc, series_size=series.size
                )
                # ... and the series is convolved with the convolution matrix
                matrix_convolved_series = conv_matrix @ series

                # the actual test is performed
                assert np.allclose(matrix_convolved_series, numpy_convolved_series), (
                    f"Differences by matrix product for Difference order {diff} with "
                    f"accuracy {acc} for series of size {size} failed."
                )


def test_stepwise_lu_banded_solve() -> None:
    """Tests the LU decomposition of a banded matrix by comparing the solution of the
    linear systems involved in Whittaker smoothing with the solution obtained by NumPy's
    ``solve``.
    """
    # the random number generator is seeded
    np.random.seed(seed=42)

    # different sizes are tested with different combinations of sub- and superdiagonals
    # counts (only small sizes can be tested sine a dense matrix is generated)
    for size in [100, 500, 1000, 5000]:
        #  random right hand side vector is generated
        b = np.random.rand(size)
        for diff in range(0, 10):
            for with_finite_check in [True, False]:
                # a finite difference matrix is generated with an updated diagonal to
                # ensure positive definiteness
                l_and_u = (diff, diff)
                d = forward_finite_diff_conv_matrix(
                    differences=diff, accuracy=1, series_size=size
                )
                a = d.T @ d + speye(size)

                # it is converted to LU banded storage ...
                ab = conv_to_lu_banded_storage(a=a, l_and_u=l_and_u)
                # ... its LU decomposition is computed ...
                lub, ipiv = lu_banded(
                    l_and_u=l_and_u,
                    ab=ab,
                    overwrite_ab=False,
                    check_finite=with_finite_check,
                )
                # ... and the linear system is solved
                x = lu_solve_banded(
                    decomposition=(l_and_u, lub, ipiv),
                    b=b,
                    check_finite=with_finite_check,
                )

                # the solution is compared to the solution obtained by NumPy's
                # solve
                np_x = np.linalg.solve(a=a.toarray(), b=b)  # type: ignore

                assert np.allclose(x, np_x), (
                    f"Banded LU decomposition for matrix of size {size} with {diff} "
                    f"sub- and superdiagonals failed."
                )


def test_lu_banded_slogdet() -> None:
    """Tests the computation of the sign and log determinant of a banded matrix from
    its LU decomposition by comparing it to NumPy's ``slogdet``.
    """

    # the random number generator is seeded
    np.random.seed(seed=42)

    # different sizes are tested with different combinations of sub- and superdiagonals
    # counts (only small sizes can be tested sine a dense matrix is generated)
    for size in [100, 500, 1000, 5000]:
        for diff in range(0, 10):
            for with_finite_check in [True, False]:
                # a finite difference matrix is generated with an updated diagonal to
                # ensure positive definiteness
                l_and_u = (diff, diff)
                d = forward_finite_diff_conv_matrix(
                    differences=diff, accuracy=1, series_size=size
                )
                a = d.T @ d + speye(size)

                # it is converted to LU banded storage ...
                ab = conv_to_lu_banded_storage(a=a, l_and_u=l_and_u)
                # ... its LU decomposition is computed ...
                lub, ipiv = lu_banded(
                    l_and_u=l_and_u,
                    ab=ab,
                    overwrite_ab=False,
                    check_finite=with_finite_check,
                )
                # ... and the sign and log determinant are computed
                sign, logabsdet = slogdet_lu_banded(
                    decomposition=(l_and_u, lub, ipiv),
                )

                # the sign and log determinant are compared to the values obtained by
                # NumPy's slogdet
                np_sign, np_logabsdet = np.linalg.slogdet(a=a.toarray())  # type: ignore

                assert np.isclose(sign, np_sign), (
                    f"Sign of log determinant for matrix of size {size} with {diff} "
                    f"sub- and superdiagonals failed."
                )
                assert np.isclose(logabsdet, np_logabsdet), (
                    f"Log determinant for matrix of size {size} with {diff} "
                    f"sub- and superdiagonals failed."
                )
