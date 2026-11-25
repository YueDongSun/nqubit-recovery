# AOD Calibration Module
from .nonlinear_model import AODCalibrationData, AODNonlinearModel, LookupTableAODModel, PolynomialAODModel
from .gp_lookup_table import GPLookupTable
from .multifrequency_coupling import (
    MultiFrequencyCouplingModel,
    MultiFrequencyCouplingNet,
    MultiFrequencyFeatures,
    build_default_features,
)

try:
    from .neural_net_correction import AODCorrectionNN, AODCorrectionNNModel
except Exception:
    AODCorrectionNN = None
    AODCorrectionNNModel = None

__all__ = [
    'AODCalibrationData',
    'AODNonlinearModel',
    'LookupTableAODModel',
    'PolynomialAODModel',
    'AODCorrectionNN',
    'AODCorrectionNNModel',
    'GPLookupTable',
    'MultiFrequencyCouplingModel',
    'MultiFrequencyCouplingNet',
    'MultiFrequencyFeatures',
    'build_default_features',
]
