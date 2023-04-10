from sklearn.utils.estimator_checks import check_estimator

from chemotools.baseline import AirPls, NonNegative
from chemotools.derivative import NorrisWilliams, SavitzkyGolay
from chemotools.normalize import MinMaxNormalize, LNormalize
from chemotools.scattering import MultiplicativeScatterCorrection, StandardNormalVariate
from chemotools.smoothing import SavitzkyGolayFilter, WhittakerSmooth


# AirPls
def test_compliance_air_pls():
    # Arrange
    transformer = AirPls()
    # Act & Assert
    check_estimator(transformer)


# LNormalize
def test_compliance_l_norm():
    # Arrange
    transformer = LNormalize()
    # Act & Assert
    check_estimator(transformer)


# MinMaxNormalize
def test_compliance_min_max_norm():
    # Arrange
    transformer = MinMaxNormalize()
    # Act & Assert
    check_estimator(transformer)


# MultiplicativeScatterCorrection
def test_compliance_multiplicative_scatter_correction():
    # Arrange
    transformer = MultiplicativeScatterCorrection()
    # Act & Assert
    check_estimator(transformer)


# NonNegative
def test_compliance_non_negative():
    # Arrange
    transformer = NonNegative()
    # Act & Assert
    check_estimator(transformer)


# NorrisWilliams
def test_compliance_norris_williams():
    # Arrange
    transformer = NorrisWilliams()
    # Act & Assert
    check_estimator(transformer)

# SavitzkyGolay
def test_compliance_savitzky_golay():
    # Arrange
    transformer = SavitzkyGolay()
    # Act & Assert
    check_estimator(transformer)


# SavitzkyGolayFilter
def test_compliance_savitzky_golay_filter():
    # Arrange
    transformer = SavitzkyGolayFilter()
    # Act & Assert
    check_estimator(transformer)


# StandardNormalVariate
def test_compliance_standard_normal_variate():
    # Arrange
    transformer = StandardNormalVariate()
    # Act & Assert
    check_estimator(transformer)


# WhittakerSmooth
def test_compliance_whittaker_smooth():
    # Arrange
    transformer = WhittakerSmooth()
    # Act & Assert
    check_estimator(transformer)
