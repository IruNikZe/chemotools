import numpy as np

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
    """Checks the generates convolution matrix for forward finite differences by
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
                # the kernel is computed
                kernel = calc_forward_diff_kernel(differences=diff, accuracy=acc)
                # and the random series is convolved with the kernel
                # NOTE: the kernel is flipped because of the way NumPy's convolve works
                numpy_convolved_series = np.convolve(
                    series, np.flip(kernel), mode="valid"
                )

                # the convolution matrix is computed
                conv_matrix = forward_finite_diff_conv_matrix(
                    differences=diff, accuracy=acc, series_size=series.size
                )
                # the series is convolved with the convolution matrix
                matrix_convolved_series = conv_matrix @ series

                # the actual test is performed
                assert np.allclose(matrix_convolved_series, numpy_convolved_series), (
                    f"Difference order {diff} with accuracy {acc} for series of size "
                    f"{size}."
                )
