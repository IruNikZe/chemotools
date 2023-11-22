import numpy as np

from chemotools._utils.finite_differences import calc_forward_diff_kernel
from tests.fixtures import reference_finite_differences


def test_forward_diff_kernel(
    reference_finite_differences: list[tuple[int, int, np.ndarray]]
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
