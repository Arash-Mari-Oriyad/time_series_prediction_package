import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import random
import sys
import os

def create_time_stamp(data, time_format, required_suffix):
    
    data.loc[:,('temporal id')] = data['temporal id'].astype(str) + required_suffix
    data.loc[:,('temporal id')] = data['temporal id'].apply(lambda x:datetime.datetime.strptime(x,time_format))
    return data

def get_target_temporal_ids(temporal_data, forecast_horizon, granularity):
    
    scale = None
    list_of_supported_formats_string_length = [4,7,10,13,16,19]
    scale_format = {'sec':'%Y/%m/%d %H:%M:%S', 'min':'%Y/%m/%d %H:%M', 'hour':'%Y/%m/%d %H', 'day':'%Y/%m/%d', 'week':'%Y/%m/%d', 'month':'%Y/%m', 'year':'%Y'}
    scale_delta = {'sec':0, 'min':0, 'hour':0, 'day':0, 'week':0, 'month':0, 'year':0}
    
    temporal_data = temporal_data.sort_values(by = ['spatial id','temporal id']).copy()
    temporal_id_instance = str(temporal_data['temporal id'].iloc[0])
    
    if len(temporal_id_instance) not in list_of_supported_formats_string_length:
        return temporal_data, None
    
    #################################### find the scale
    
    # try catch is used to detect non-integrated temporal id format in the event of an error and return None as scale
    try:
        if len(temporal_id_instance) == 4:
            scale = 'year'
            temporal_data = create_time_stamp(temporal_data.copy(), '%Y/%m/%d', '/01/01')

        elif len(temporal_id_instance) == 7:
            scale = 'month'
            temporal_data = create_time_stamp(temporal_data.copy(), '%Y/%m/%d', '/01')

        elif len(temporal_id_instance) == 10:

            temporal_data = create_time_stamp(temporal_data.copy(), '%Y/%m/%d', '')

            first_temporal_id_instance = temporal_data['temporal id'].iloc[0]
            second_temporal_id_instance = temporal_data['temporal id'].iloc[1]

            delta = second_temporal_id_instance - first_temporal_id_instance
            if delta.days == 1:
                scale = 'day'
            elif delta.days == 7:
                scale = 'week'
                
        elif len(temporal_id_instance) == 13:
            scale = 'hour'
            temporal_data = create_time_stamp(temporal_data.copy(), '%Y/%m/%d %H:%M:%S', ':00:00')

        elif len(temporal_id_instance) == 16:
            scale = 'min'
            temporal_data = create_time_stamp(temporal_data.copy(), '%Y/%m/%d %H:%M:%S', ':00')

        elif len(temporal_id_instance) == 19:
            scale = 'sec'
            temporal_data = create_time_stamp(temporal_data.copy(), '%Y/%m/%d %H:%M:%S', '')
            
    except ValueError:
            return temporal_data, None

    ##################################### get the target time point
    
    # The time point of target variable is granularity*(forecast_horizon) units ahead of the data current temporal id
    scale_delta[scale] = granularity*(forecast_horizon)
    timestamp_delta = relativedelta(years = scale_delta['year'], months = scale_delta['month'], weeks = scale_delta['week'], days = scale_delta['day'], hours = scale_delta['hour'], minutes = scale_delta['min'], seconds = scale_delta['sec'])

    temporal_data['temporal id'] = temporal_data['temporal id'].apply(lambda x: datetime.datetime.strftime((x + timestamp_delta),scale_format[scale]))

    return temporal_data, scale

def create_plot(df, forecast_horizon, granularity, spatial_ids, save_address, plot_type, test_point):
    
    mpl.style.use('default')
    df, scale = get_target_temporal_ids(df, forecast_horizon, granularity)
    df = df.sort_values(by = ['spatial id','temporal id'])
    
    if scale is not None:
        x_axis_label = 'Target time point'
    else:
        x_axis_label = 'Predictive time point'
    
    if spatial_ids is None:
        spatial_ids = list(random.sample(list(df['spatial id']),1))
        
    temporal_ids = list(df['temporal id'].unique())
    
    for spatial_id in spatial_ids:
        stage = 'training' if plot_type == 'test' else 'forecast'
        
        if test_point is not None:
            save_file_name = '{0}spatial id = {1} {2} stage for test point #{3}.pdf'.format(save_address, spatial_id, stage, test_point+1)
        else:
            save_file_name = '{0}spatial id = {1} {2} stage.pdf'.format(save_address, spatial_id, stage)
        
        plt.rc('font', size=60)
        number_of_temporal_ids = len(temporal_ids) + 2
        fig, ax = plt.subplots()
        
        # add the curve of real values of the target variable
        temp_df = df[df['spatial id'] == spatial_id]
        plt.plot(list(temp_df['temporal id']),list(temp_df['real']),label='Real values', marker = 'o', markersize=20, linewidth=3.0)
        
        # add the curve of predicted values of the target variable in the training, validation and testing set
        if plot_type != 'future':
            temp_train_df = temp_df[temp_df['sort'] == 'train']
            plt.plot(list(temp_train_df['temporal id']),list(temp_train_df['prediction']),label='Training set predicted values', marker = 'o', markersize=20, linewidth=3.0)
            temp_val_df = temp_df[temp_df['sort'] == 'validation']
            plt.plot(list(temp_val_df['temporal id']),list(temp_val_df['prediction']),label='validation set predicted values', marker = 'o', markersize=20, linewidth=3.0)
            temp_test_df = temp_df[temp_df['sort'] == 'test']
            plt.plot(list(temp_test_df['temporal id']),list(temp_test_df['prediction']),label='Testing set predicted values', marker = 'o', markersize=20, linewidth=3.0)

        if plot_type == 'future':
            temp_test_df = temp_df[temp_df['sort'] == 'test']
            plt.plot(list(temp_test_df['temporal id']),list(temp_test_df['prediction']),label='Predicted values', marker = 'o', markersize=20, linewidth=3.0)
        
        plt.ylabel('Target')
        plt.xlabel(x_axis_label,labelpad = 20)
        plt.legend()
        plt.xticks(rotation=90)
        plt.grid()
        
        # set the size of plot base on number of temporal units and lable fonts
        plt.gca().margins(x=0.002)
        plt.gcf().canvas.draw()
        tl = plt.gca().get_xticklabels()
        maxsize = max([t.get_window_extent().width for t in tl])
        inch_margin = 0.5 # inch margin
        xtick_size = maxsize/plt.gcf().dpi*number_of_temporal_ids+inch_margin
        margin = inch_margin/plt.gcf().get_size_inches()[0]

        plt.gcf().subplots_adjust(left=margin, right=1.-margin)
        plt.gcf().set_size_inches(xtick_size, plt.gcf().get_size_inches()[1]*5)
        
        plt.tight_layout()
        plt.margins(x = 0.01)
        
        try:
            if not os.path.exists(save_address):
                os.makedirs(save_address)
            plt.savefig(save_file_name, bbox_inches='tight')
            plt.close()
        except FileNotFoundError:
                print("The address '{0}' is not valid.".format(save_address))
                
def plot_predictions(data, test_type = 'whole-as-one', forecast_horizon = 1, plot_type = 'test', granularity = 1,
                        spatial_ids = None):
    
    validation_dir = './prediction/validation process/'
    testing_dir = './prediction/test process/'
    future_dir = './prediction/future prediction/'
    needed_columns = ['temporal id', 'spatial id','real','prediction','sort']

    path = validation_dir
    files = [f for f in listdir(path) if isfile(join(path, f))]
    prefix = 'validation prediction forecast horizon = {0}, T ='.format(forecast_horizon)
    files = [file for file in files if file.startswith(prefix)]
    file_temporal_units = [int(file.split('T = ')[1][:-4]) for file in files]
    file_temporal_units.sort()
    
    if plot_type == 'test':
        address = testing_dir + 'test'
    elif plot_type == 'future':
        address = future_dir + 'future'

    if test_type == 'whole-as-one':

        if plot_type == 'test':
            temporal_units_number = file_temporal_units[0]
        elif plot_type == 'future':
            temporal_units_number = file_temporal_units[-1]

        test_df = pd.read_csv(address + ' prediction forecast horizon = {0}.csv'.format(forecast_horizon))
        selected_model = list(test_df['model name'].unique())[0]
        test_df = test_df.assign(sort = 'test')
        train_df = pd.read_csv(validation_dir + 'training prediction forecast horizon = {0}, T = {1}.csv'.format(forecast_horizon,temporal_units_number))
        train_df = train_df[train_df['model name'] == selected_model]
        train_df = train_df.assign(sort = 'train')
        validation_df = pd.read_csv(validation_dir + 'validation prediction forecast horizon = {0}, T = {1}.csv'.format(forecast_horizon,temporal_units_number))
        validation_df = validation_df[validation_df['model name'] == selected_model]
        validation_df = validation_df.assign(sort = 'validation')
        gap_df = data.rename(columns = {'Normal target':'real'})
        gap_df = gap_df.assign(prediction = np.NaN)
        gap_df = gap_df.assign(sort = 'gap')
        gap_df = gap_df[(gap_df['temporal id'] < test_df['temporal id'].min()) & (gap_df['temporal id'] > train_df.append(validation_df)['temporal id'].max())]

        all_df = train_df[needed_columns].append(validation_df[needed_columns]).append(gap_df[needed_columns]).append(test_df[needed_columns])
        create_plot(df = all_df, forecast_horizon = forecast_horizon, granularity = granularity, spatial_ids = None, 
                        save_address = './plots/', plot_type = plot_type, test_point = None)

    if test_type == 'one-by-one':    
        
        test_point_number = len(file_temporal_units)-1
        all_test_points_df = pd.read_csv(address + ' prediction forecast horizon = {0}.csv'.format(forecast_horizon))
        test_temporal_units = all_test_points_df['temporal id'].unique()
        test_temporal_units.sort()
        for test_point in range(test_point_number):
            test_df = all_test_points_df[all_test_points_df['temporal id'] == test_temporal_units[test_point]]
            selected_model = list(test_df['model name'].unique())[0]
            test_df = test_df.assign(sort='test')
            if plot_type == 'test':
                temporal_units_number = file_temporal_units[test_point]
            if plot_type == 'future':
                temporal_units_number = file_temporal_units[-1]

            train_df = pd.read_csv(validation_dir + 'training prediction forecast horizon = {0}, T = {1}.csv'.format(forecast_horizon,temporal_units_number))
            train_df = train_df[train_df['model name'] == selected_model]
            train_df = train_df.assign(sort = 'train')
            validation_df = pd.read_csv(validation_dir + 'validation prediction forecast horizon = {0}, T = {1}.csv'.format(forecast_horizon,temporal_units_number))
            validation_df = validation_df[validation_df['model name'] == selected_model]
            validation_df = validation_df.assign(sort = 'validation')
            gap_df = data.rename(columns = {'Normal target':'real'})
            gap_df = gap_df.assign(prediction = np.NaN)
            gap_df = gap_df.assign(sort = 'gap')
            gap_df = gap_df[(gap_df['temporal id'] < test_df['temporal id'].min()) & (gap_df['temporal id'] > train_df.append(validation_df)['temporal id'].max())]


            all_df = train_df[needed_columns].append(validation_df[needed_columns]).append(gap_df[needed_columns]).append(test_df[needed_columns])
            create_plot(df = all_df, forecast_horizon = forecast_horizon, granularity = granularity, spatial_ids = None, 
                        save_address = './plots/', plot_type = plot_type, test_point = test_point)
