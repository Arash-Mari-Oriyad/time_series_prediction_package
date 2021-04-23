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
