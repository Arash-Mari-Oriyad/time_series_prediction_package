import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import random
import sys
import os
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors

# merge the cells in matplotlib table
def mergecells(table, cells):
    cells_array = [np.asarray(c) for c in cells]
    h = np.array([cells_array[i+1][0] - cells_array[i][0] for i in range(len(cells_array) - 1)])
    v = np.array([cells_array[i+1][1] - cells_array[i][1] for i in range(len(cells_array) - 1)])

    # if it's a horizontal merge, all values for `h` are 0
    if not np.any(h):
        # sort by horizontal coord
        cells = np.array(sorted(list(cells), key=lambda v: v[1]))
        edges = ['BTL'] + ['BT' for i in range(len(cells) - 2)] + ['BTR']
    elif not np.any(v):
        cells = np.array(sorted(list(cells), key=lambda h: h[0]))
        edges = ['TRL'] + ['RL' for i in range(len(cells) - 2)] + ['BRL']
    else:
        raise ValueError("Only horizontal and vertical merges allowed")

    for cell, e in zip(cells, edges):
        table[cell[0], cell[1]].visible_edges = e
        
    txts = [table[cell[0], cell[1]].get_text() for cell in cells]
    tpos = [np.array(t.get_position()) for t in txts]

    # transpose the text of the left cell
    trans = (tpos[-1] - tpos[0])/2
    # didn't had to check for ha because I only want ha='center'
    txts[0].set_transform(mpl.transforms.Affine2D().translate(*trans))
    for txt in txts[1:]:
        txt.set_visible(False)
        
# find the consecutive cells with same values and merge them
def merge_same_values(row_number,column,offset,the_table):
    values = [the_table.get_celld()[(i,column)].get_text().get_text() for i in range(offset,row_number+1)]
    current_value = values[0]
    merge_list = []
    for index,value in enumerate(values):
        if value == current_value:
            merge_list = merge_list+[(index+offset,column)]
        else:
            current_value = value
            mergecells(the_table, merge_list)
            merge_list = [(index+offset,column)]
            
    mergecells(the_table, merge_list)
    return the_table

# find the consecutive cells with same values and colours them with the same color
def color_same_values(row_number,column_number,column,offset,the_table):
    
    colors = ['#008e99','#424242','#ba4c4d','#858585','#acc269','#7255b2']
    
    # find cells with same text
    values = [the_table.get_celld()[(i,column)].get_text().get_text() for i in range(offset,row_number+1)]
    models = np.unique(values)
    
    color_list = colors * ((len(models)//6)+1) 
    
    color_dict = {model:color_list[i] for i , model in enumerate(models)}
        
    current_value = values[0]
    for index,value in enumerate(values):
        if value == current_value:
            for col in range(column,column_number):
                the_table.get_celld()[(index+offset,col)].get_text().set_color(color_dict[value])
        else:
            current_value = value
            for col in range(column,column_number):
                the_table.get_celld()[(index+offset,col)].get_text().set_color(color_dict[value])
            
    return the_table

# prepare input df by counting features and rounding performance values
def prepare_df(df,base_columns,measure_names,performance_benchmark,test_type,plot_type):
    
    if test_type == 'whole-as-one':
        unique_columns = ['model name', 'history length']
    if test_type == 'one-by-one':
        unique_columns = ['Test point','model name','history length']
        
    df['feature or covariate set'] = df['feature or covariate set'].apply(lambda x:len(x.split(',')))
    max_performance_df = df.groupby(unique_columns)[[performance_benchmark]].max().reset_index().rename(columns = {performance_benchmark:'best '+performance_benchmark})
    df = pd.merge(df,max_performance_df)
    df = df[df[performance_benchmark] == df['best '+performance_benchmark]].drop_duplicates(subset = unique_columns)
    df = df[base_columns + measure_names]
    if plot_type == 'table':
        for measure in measure_names:
            df[measure] = df[measure].apply(lambda x:np.round(x,2))
            df.loc[df[measure]>99999,measure] = df.loc[df[measure]>99999,measure].apply(lambda x:'{:.5E}'.format(x))
    return df

# make tables and save into pdf
def plot_table(train_df,validation_df,test_df,test_type,performance_benchmark,test_point_number):
    
    if test_type == 'whole-as-one':
        base_columns = ['Dataset', 'model name', 'history length', 'feature or covariate set']
    if test_type == 'one-by-one':
        base_columns = ['Test point', 'Dataset', 'model name', 'history length', 'feature or covariate set']
    measure_names = list(filter(lambda x: x not in base_columns, test_df.columns))
    models_number = len(train_df['model name'].unique())
    max_history = len(train_df['history length'].unique())

    test_df = prepare_df(test_df,base_columns,measure_names,performance_benchmark,test_type,'table')
    train_df = prepare_df(train_df,base_columns,measure_names,performance_benchmark,test_type,'table')
    validation_df = prepare_df(validation_df,base_columns,measure_names,performance_benchmark,test_type,'table')

    if test_type == 'whole-as-one':
        table = test_df.values.tolist()
        table = table + [['Training','Models','History length','Feature set'] + measure_names]
        table = table + train_df.values.tolist()
        table = table + validation_df.values.tolist()
        colWidths = [0.2, 0.2, 0.3, 0.25] + [0.2]*len(measure_names)
        row_number = len(table)#2*models_number*max_history

    if test_type == 'one-by-one':
        table = test_df.values.tolist()
        table = table + [['1','Training','Models','History length','Feature set'] + measure_names]
        for test_point in range(1,test_point_number+1):
            table = table + train_df[train_df['Test point'] == test_point].values.tolist()
            table = table + validation_df[validation_df['Test point'] == test_point].values.tolist()
        colWidths = [0.2, 0.2, 0.2, 0.3, 0.25] + [0.2]*len(measure_names)
        row_number = len(table)#2*models_number*max_history*test_point_number
    
    # calculate number of pages needed
    npages = row_number // 300
    if row_number % 300 > 0:
        npages += 1
    
    pdf = PdfPages('./performance/performance summary.pdf')
    
    for page in range(npages):
        fig = plt.figure()
        plt.tight_layout()
        ax = fig.gca()
        ax.axis('tight')
        ax.axis('off')
        temp_table = table[page*300:(page+1)*300]
        
        if test_type == 'whole-as-one' and page == 0:
            colLabels = ['Dataset','Best model','Optimal history length','Best feature set'] + ['Best ' + item for item in measure_names]
        elif test_type == 'whole-as-one' and page > 0:
            colLabels = ['Dataset','Models','History length','Feature set'] + [item for item in measure_names]
        
        if test_type == 'one-by-one' and page == 0:
            colLabels = ['Test point', 'Dataset','Best model','Optimal history length','Best feature set'] + ['Best ' + item for item in measure_names]
        elif test_type == 'one-by-one' and page > 0:
            colLabels = ['Test point', 'Dataset','Best model','History length','Feature set'] + [item for item in measure_names]
            
        the_table = ax.table(cellText=temp_table, colLabels=colLabels, colWidths=colWidths, loc='center', cellLoc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(9)
        the_table.scale(1.5, 1.5)
        fig.canvas.draw()

        if test_type == 'whole-as-one':
            
            # merge the cells containing dataset name in first column
            offset = 2 if page == 0 else 1
            the_table = merge_same_values(row_number = len(temp_table),
                                  column = 0, offset = offset, the_table = the_table)
            # merge the cells containing model name in second column 
            offset = 3 if page == 0 else 1
            the_table = color_same_values(row_number = len(temp_table), column_number = len(colLabels),
                                  column = 1, offset = offset, the_table = the_table)
            the_table = merge_same_values(row_number = len(temp_table),
                                  column = 1, offset = offset, the_table = the_table)
        if test_type == 'one-by-one':
            # merge the cells containing test point number in first column
            offset = 2 if page == 0 else 1
            the_table = merge_same_values(row_number = len(temp_table),
                                          column = 0, offset = offset, the_table = the_table)
            # merge the cells containing dataset name in second column
            the_table = merge_same_values(row_number = len(temp_table),
                                          column = 1, offset = offset, the_table = the_table)
            # merge the cells containing model name in third column
            offset = 3 if page == 0 else 1
            the_table = color_same_values(row_number = len(temp_table), column_number = len(colLabels),
                                  column = 2, offset = offset, the_table = the_table)
            the_table = merge_same_values(row_number = len(temp_table),
                                          column = 2, offset = offset, the_table = the_table)

        # colour header
        for i in range(len(colLabels)):
            the_table.get_celld()[(0,i)].set_facecolor('lightgray')#'#40466e'
            the_table.get_celld()[(0,i)].set_text_props(weight='bold')#, color='w')
            if (page == 0) and ((test_type == 'whole-as-one' and i>0) or (test_type == 'one-by-one' and i>1)):
                the_table.get_celld()[(2,i)].set_facecolor('lightgray')#'#40466e'
                the_table.get_celld()[(2,i)].set_text_props(weight='bold')#, color='w')

        pdf.savefig(fig , dpi=300, bbox_inches='tight', pad_inches=1)
        plt.close()

    pdf.close()
    
# make barplots and save into pdf

def plot_bar(train_df,validation_df,test_type,performance_benchmark,test_point_number):
    
    if test_type == 'whole-as-one':
        base_columns = ['Dataset', 'model name', 'history length', 'feature or covariate set']
    if test_type == 'one-by-one':
        base_columns = ['Test point', 'Dataset', 'model name', 'history length', 'feature or covariate set']
    measure_names = list(filter(lambda x: x not in base_columns, train_df.columns))
    models = train_df['model name'].unique()
    models_number = len(models)
    max_history = len(train_df['history length'].unique())
    
    colors = ['#008e99','#424242','#ba4c4d','#858585','#acc269','#7255b2']
    color_list = colors * ((len(models)//6)+1) 
    color_dict = {model:color_list[i] for i , model in enumerate(models)}
        
    train_df = prepare_df(train_df,base_columns,measure_names,performance_benchmark,test_type,'bar')
    validation_df = prepare_df(validation_df,base_columns,measure_names,performance_benchmark,test_type,'bar')
    
    if test_type == 'whole-as-one':
        test_point_number = 1
    
    models_per_plot = 6
    # calculate number of pages needed
    npages = models_number // models_per_plot
    if models_number % models_per_plot > 0:
        npages += 1
        
    barWidth = 0.015     
    pos = 0.195*np.arange(max_history)
    log_flag = False
    
    # initialize pdf
    pdf = PdfPages('./performance/performance bar plots.pdf')
    
    for test_point in range(1,test_point_number+1):
        
        for page in range(npages):
            
            mpl.rcParams.update({'font.size': 12})
            fig, ax = plt.subplots()
            if test_type == 'one-by-one':
                plt.title('Test point number {0}'.format(test_point),fontweight='bold')
            plt.tight_layout()
            
            current_models = train_df['model name'].unique()[page*models_per_plot:(page+1)*models_per_plot]
                
            for index,model in enumerate(current_models):
                
                # Set position of bar on X axis
                current_train_pos=[x + (2*index)*barWidth for x in pos]
                current_validation_pos=[x + (2*index+1)*barWidth for x in pos]
                
                if test_type == 'whole-as-one':
                    train_performance = list(train_df.loc[train_df['model name'] == model,performance_benchmark])
                    validation_performance = list(validation_df.loc[validation_df['model name'] == model,performance_benchmark])
                
                if test_type == 'one-by-one':
                    train_performance = list(train_df.loc[(train_df['Test point'] == test_point)&(train_df['model name'] == model),performance_benchmark])
                    validation_performance = list(validation_df.loc[(validation_df['Test point'] == test_point)&(validation_df['model name'] == model),performance_benchmark])
                
                if max(train_performance+validation_performance)-min(train_performance+validation_performance)>10000:
                    log_flag = True
                    
                
                # Make the plot
                ax.bar(current_train_pos, train_performance, color=color_dict[model], width=barWidth, edgecolor='white',hatch='//////')
                ax.bar(current_validation_pos, validation_performance, color=color_dict[model], width=barWidth, edgecolor='white', label=model)

                plt.xlabel('History Length', fontweight='bold')
                plt.ylabel("AUC",fontweight='bold')
                plt.xticks([r + (len(current_models)-0.5)*barWidth for r in pos], [f'HL={history}' for history in range (1,max_history+1)])

                
                train_patch = mpatches.Patch(edgecolor='gray', facecolor='w', label='Training', hatch='//////')
                validation_patch = mpatches.Patch(color='gray', label='Validation')
                leg1 = plt.legend(handles=[train_patch,validation_patch], loc='upper right')
                ax.add_artist(leg1)
    
                # Create legend & Show graphic
                plt.legend(loc='lower right')
                
                if log_flag == True:
                    plt.yscale('log')
            
            
            if max_history>5:
                plt.gcf().set_size_inches(plt.gcf().get_size_inches()[0]*(max_history/5), plt.gcf().get_size_inches()[1])
            else:
                plt.gcf().set_size_inches(plt.gcf().get_size_inches()[0]*(1.5), plt.gcf().get_size_inches()[1])
            pdf.savefig(fig , dpi=300, bbox_inches='tight', pad_inches=1)
            plt.close()

    pdf.close()

# save summary of performance reports

def performance_summary(forecast_horizon,test_type,performance_benchmark):

    validation_dir = './performance/validation process/'
    testing_dir = './performance/test process/'


    path = validation_dir
    files = [f for f in listdir(path) if isfile(join(path, f))]
    prefix = 'validation performance report forecast horizon = {0}, T ='.format(forecast_horizon)
    files = [file for file in files if file.startswith(prefix)]
    file_temporal_units = [int(file.split('T = ')[1][:-4]) for file in files]
    file_temporal_units.sort()

    if test_type == 'whole-as-one':

        temporal_units_number = file_temporal_units[0]

        test_df = pd.read_csv(testing_dir + 'test performance report forecast horizon = {0}.csv'.format(forecast_horizon))
        test_df.insert(0, 'Dataset', 'Test')

        train_df = pd.read_csv(validation_dir + 'training performance report forecast horizon = {0}, T = {1}.csv'.format(forecast_horizon,temporal_units_number))
        train_df.insert(0, 'Dataset', 'Training')

        validation_df = pd.read_csv(validation_dir + 'validation performance report forecast horizon = {0}, T = {1}.csv'.format(forecast_horizon,temporal_units_number))
        validation_df.insert(0, 'Dataset', 'validation')

        plot_table(train_df,validation_df,test_df,test_type,performance_benchmark,None)

    elif test_type == 'one-by-one':

        test_point_number = len(file_temporal_units)    
        test_df = pd.read_csv(testing_dir + 'test performance report forecast horizon = {0}.csv'.format(forecast_horizon))
        test_df.insert(0, 'Test point', 'Overall')
        test_df.insert(1, 'Dataset', 'Test')


        train_df = pd.DataFrame()
        validation_df = pd.DataFrame()
        for test_point in range(1,test_point_number+1):

            temporal_units_number = file_temporal_units[test_point-1]

            temp_train_df = pd.read_csv(validation_dir + 'training performance report forecast horizon = {0}, T = {1}.csv'.format(forecast_horizon,temporal_units_number))
            temp_train_df.insert(0, 'Test point', test_point)
            temp_train_df.insert(1, 'Dataset', 'Training')
            train_df = train_df.append(temp_train_df)

            temp_validation_df = pd.read_csv(validation_dir + 'validation performance report forecast horizon = {0}, T = {1}.csv'.format(forecast_horizon,temporal_units_number))
            temp_validation_df.insert(0, 'Test point', test_point)
            temp_validation_df.insert(1, 'Dataset', 'validation')
            validation_df = validation_df.append(temp_validation_df)

        plot_table(train_df,validation_df,test_df,test_type,performance_benchmark,test_point_number)


# plot the performance bars for training and validation
def performance_bar_plot(forecast_horizon,test_type,performance_benchmark):

    validation_dir = './performance/validation process/'
    testing_dir = './performance/test process/'


    path = validation_dir
    files = [f for f in listdir(path) if isfile(join(path, f))]
    prefix = 'validation performance report forecast horizon = {0}, T ='.format(forecast_horizon)
    files = [file for file in files if file.startswith(prefix)]
    file_temporal_units = [int(file.split('T = ')[1][:-4]) for file in files]
    file_temporal_units.sort()

    if test_type == 'whole-as-one':

        temporal_units_number = file_temporal_units[0]

        train_df = pd.read_csv(validation_dir + 'training performance report forecast horizon = {0}, T = {1}.csv'.format(forecast_horizon,temporal_units_number))
        train_df.insert(0, 'Dataset', 'Training')

        validation_df = pd.read_csv(validation_dir + 'validation performance report forecast horizon = {0}, T = {1}.csv'.format(forecast_horizon,temporal_units_number))
        validation_df.insert(0, 'Dataset', 'validation')

        plot_bar(train_df,validation_df,test_type,performance_benchmark,None)

    elif test_type == 'one-by-one':

        test_point_number = len(file_temporal_units)    
        test_df = pd.read_csv(testing_dir + 'test performance report forecast horizon = {0}.csv'.format(forecast_horizon))
        test_df.insert(0, 'Test point', 'Overall')
        test_df.insert(1, 'Dataset', 'Test')


        train_df = pd.DataFrame()
        validation_df = pd.DataFrame()
        for test_point in range(1,test_point_number+1):

            temporal_units_number = file_temporal_units[test_point-1]

            temp_train_df = pd.read_csv(validation_dir + 'training performance report forecast horizon = {0}, T = {1}.csv'.format(forecast_horizon,temporal_units_number))
            temp_train_df.insert(0, 'Test point', test_point)
            temp_train_df.insert(1, 'Dataset', 'Training')
            train_df = train_df.append(temp_train_df)

            temp_validation_df = pd.read_csv(validation_dir + 'validation performance report forecast horizon = {0}, T = {1}.csv'.format(forecast_horizon,temporal_units_number))
            temp_validation_df.insert(0, 'Test point', test_point)
            temp_validation_df.insert(1, 'Dataset', 'validation')
            validation_df = validation_df.append(temp_validation_df)

        plot_bar(train_df,validation_df,test_type,performance_benchmark,test_point_number)
