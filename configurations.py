NON_FEATURE_COLUMNS_NAMES = ['spatial id', 'temporal id']

TARGET_COLUMN_NAME = 'Target'

NORMAL_TARGET_COLUMN_NAME = 'Normal target'

BASIC_TARGET_COLUMN_NAMES = ['Target (normal)',
                             'Target (augmented on z units - normal)',
                             'Target (differential)',
                             'Target (augmented on z units - differential)',
                             'Target (moving average on x units)',
                             'Target (augmented on z units - moving average on x units)',
                             'Target (cumulative)',
                             'Target (augmented on z units - cumulative)']

TEST_TYPES = ['one-by-one', 'whole-as-one']

TARGET_MODES = ['normal', 'cumulative', 'differential', 'moving_average']

RANKING_METHODS = ['mRMR', 'CORRELATION']

FEATURE_SELECTION_TYPES = ['covariate', 'feature']

PRE_DEFINED_MODELS = ['nn', 'knn', 'glm', 'gbm']

MODEL_TYPES = ['regression', 'classification']

PERFORMANCE_MEASURES = ['MAE', 'MAPE', 'MASE', 'MSE', 'R2_score']

PERFORMANCE_BENCHMARKS = ['MAPE']

FEATURE_SCALERS = ['logarithmic', 'normalize', 'standardize', None]

TARGET_SCALERS = ['logarithmic', 'normalize', 'standardize', None]

SPLITTING_TYPES = ['training-validation', 'cross-validation']

VERBOSE_OPTIONS = [0, 1, 2]
