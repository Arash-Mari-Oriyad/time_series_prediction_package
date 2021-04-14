import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta


############################################# manipulating temporal id's for ploting the target real and predicted values

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


########################################## plot real and predicted values of target

def target_prediction_plot(train_real_values, test_real_values, train_predicted_values, test_predicted_values, plot_type,
                           forecast_horizon, save_address, granularity = 1, spatial_ids = None):
    
    '''
    train_real : a data frame of spatial_id, temoral id and target variable real values in the train set
    test_real : a data frame of spatial_id, temoral id and target variable real values in the test set
    train_predicted : a list of predicted values of target variable in the train set
    test_predicted : a list of predicted values of target variable in the test set
    plot_type : 'future' or 'train_test'
    spatial_ids : list of spatial ids to plot or None
    save_address : address to save the plots
    '''
    
    train_data = train_real_values.copy()
    train_data['prediction'] = list(train_predicted_values)
    train_data, scale = get_target_temporal_ids(train_data, forecast_horizon, granularity)
    train_data = train_data.sort_values(by = ['spatial id','temporal id'])

    test_data = test_real_values.copy()
    test_data['prediction'] = list(test_predicted_values)
    test_data, scale = get_target_temporal_ids(test_data, forecast_horizon, granularity)
    test_data = test_data.sort_values(by = ['spatial id','temporal id'])

    
    all_data = train_data.append(test_data).sort_values(by = ['spatial id','temporal id'])

    # in the case of non integrated temporal id format target time point can not be detected
    # and the current temporal ids must be shown in the plot
    if scale is not None:
        x_axis_label = 'Target time point'
    else:
        x_axis_label = 'Predictive time point'
    
    if spatial_ids is None:
        spatial_ids = list(random.sample(list(train_data['spatial id']),1))
        
    temporal_ids = list(train_data['temporal id'].unique()) + list(test_data['temporal id'].unique())
    
    for spatial_id in spatial_ids:
        plt.rc('font', size=60)
        number_of_temporal_ids = len(temporal_ids) + 2
        fig, ax = plt.subplots()
        
        # add the curve of real values of the target variable
        temp_all_data = all_data[all_data['spatial id'] == spatial_id]
        plt.plot(list(temp_all_data['temporal id']),list(temp_all_data['Target']),label='Real values', marker = 'o', markersize=20, linewidth=3.0)
        
        # add the curve of predicted values of the target variable in the training set
        if plot_type != 'future':
            temp_train_data = train_data[train_data['spatial id'] == spatial_id]
            plt.plot(list(temp_train_data['temporal id']),list(temp_train_data['prediction']),label='Training set predicted values', marker = 'o', markersize=20, linewidth=3.0)
        
        # add the curve of predicted values of the target variable in the testing set
        if plot_type == 'future': label = 'Predicted values'
        else: label = 'Testing set predicted values'
        temp_test_data = test_data[test_data['spatial id'] == spatial_id]
        plt.plot(list(temp_test_data['temporal id']),list(temp_test_data['prediction']),label=label, marker = 'o', markersize=20, linewidth=3.0)
        
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
        
        try:
            if not os.path.exists(save_address):
                os.makedirs(save_address)
            plt.savefig('{0}spatial id = {1} {2} real prediction values.png'.format(save_address, spatial_id, plot_type), bbox_inches='tight')
            plt.close()
        except FileNotFoundError:
                print("The address '{0}' is not valid.".format(save_address))

                