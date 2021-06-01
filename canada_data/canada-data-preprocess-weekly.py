import pandas as pd
from data_preprocessing import data_preprocess

data_address = './'
target_name = 'Daily Deaths'
forecast_horizon = 4
max_history = 5

data = pd.read_csv(data_address + 'Canada_data28thMay.csv')
data.loc[:, 'country_code'] = 1
data['Date'] = data['Date'].apply(lambda x: x.replace('-', '/'))

temporal_covariates = [' Daily Cases', 'Daily Deaths',
                       'retail_and_recreation_percent_change_from_baseline',
                       'grocery_and_pharmacy_percent_change_from_baseline',
                       'parks_percent_change_from_baseline',
                       'transit_stations_percent_change_from_baseline',
                       'residential_percent_change_from_baseline ']

column_identifier = {'spatial id level 1': 'country_code', 'temporal id': 'Date',
                     'temporal covariates': temporal_covariates, 'target': target_name}

history_length = {key: max_history for key in temporal_covariates}

historical_data_list = data_preprocess(data=data.copy(),
                                       forecast_horizon=forecast_horizon,
                                       history_length=history_length,
                                       column_identifier=column_identifier,
                                       spatial_scale_table=None,
                                       spatial_scale_level=1,
                                       temporal_scale_level=2,
                                       target_mode='normal',
                                       imputation=False,
                                       aggregation_mode='mean',
                                       augmentation=False,
                                       futuristic_covariates=None,
                                       future_data_table=None,
                                       save_address='',  # <------------ save address
                                       verbose=1)
