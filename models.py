import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso
import statsmodels.api as sm
from sklearn import svm
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from pexecute.process import ProcessLoom
import time
from sys import argv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.callbacks import EarlyStopping
import os
from numpy.random import seed
seed(1)
tf.random.set_seed(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


####################################################### GBM: Gradient Boosting Regressor
def GBM(X_train, X_test, y_train, user_params, verbose):

    parameters = {'loss':'least_squares', 'learning_rate':0.1, 'max_iter':100, 'max_leaf_nodes':31, 'max_depth':None,
                'min_samples_leaf':20, 'l2_regularization':0.0, 'max_bins':255, #'categorical_features':None,
                'monotonic_cst':None, 'warm_start':False, 'early_stopping':'auto', 'scoring':'loss',
                'validation_fraction':0.1, 'n_iter_no_change':10, 'tol':1e-07, 'verbose':0, 'random_state':None}
    
    if user_params is not None:
        for key in parameters.keys():
            if key in user_params.keys():
                parameters[key] = user_params[key]
                
    GradientBoostingRegressorObject = HistGradientBoostingRegressor(**parameters)

    GradientBoostingRegressorObject.fit(X_train, y_train)
    y_prediction = GradientBoostingRegressorObject.predict(X_test)
    y_prediction_train = GradientBoostingRegressorObject.predict(X_train)


    return np.array(y_prediction).ravel(), np.array(y_prediction_train).ravel(), GradientBoostingRegressorObject


###################################################### GLM: Generalized Linear Model, we use Lasso
def GLM(X_train, X_test, y_train, user_params, verbose):
    
    parameters = {'alpha':1.0, 'l1_ratio':0.5, 'fit_intercept':True, 'normalize':False, 'precompute':False,
                  'max_iter':1000, 'copy_X':True, 'tol':0.0001, 'warm_start':False, 'positive':False, 'random_state':1,
                  'selection':'cyclic'}
    
    if user_params is not None:
        for key in parameters.keys():
            if key in user_params.keys():
                parameters[key] = user_params[key]
    
    GLM_Model = ElasticNet(**parameters)
    GLM_Model.fit(X_train, y_train)
    y_prediction = GLM_Model.predict(X_test)
    y_prediction_train = GLM_Model.predict(X_train)
    
    if verbose == 1:
        print('GLM coef: ', GLM_Model.coef_)

    return np.array(y_prediction).ravel(), np.array(y_prediction_train).ravel(), GLM_Model


######################################################### KNN: K-Nearest Neighbors
def KNN(X_train, X_test, y_train, user_params, verbose):
    
    parameters = {'weights':'uniform', 'algorithm':'auto', 'leaf_size':30, 'p':2, 'metric':'minkowski',
                  'metric_params':None, 'n_jobs':None}
    
    if user_params is not None:
        for key in parameters.keys():
            if key in user_params.keys():
                parameters[key] = user_params[key]
                
    # if user does not specify the K parameter or specified value is too large, the best k will be obtained using a grid search
    if (user_params is not None) and ('n_neighbors' in user_params.keys()) and (user_params['n_neighbors']<len(X_train)):
            K = user_params['n_neighbors']
    else:
        KNeighborsRegressorObject = KNeighborsRegressor()
        # Grid search over different Ks to choose the best one
        neighbors=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,20 ,40 ,60 ,80, 100, 120, 140, 160, 180, 200])
        neighbors=neighbors[neighbors<len(X_train)*(4/5)] #4/5 of samples is used as train when cv=5
        grid_parameters = {'n_neighbors': neighbors}
        GridSearchOnKs = GridSearchCV(KNeighborsRegressorObject, grid_parameters, cv=5)
        GridSearchOnKs.fit(X_train, y_train)
        best_K = GridSearchOnKs.best_params_
        
        print("Warning: The number of neighbors for KNN algorithm is not specified or is too large for input data shape.")
        print("The number of neighbors will be set to the best number of neighbors obtained by grid search in the range [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,20 ,40 ,60 ,80, 100, 120, 140, 160, 180, 200]")
    
        if verbose == 1:
            print('best k:', best_K['n_neighbors'])
            
        K = best_K['n_neighbors']
        
    KNN_Model = KNeighborsRegressor(n_neighbors=K, **parameters)
    KNN_Model.fit(X_train, y_train)
    y_prediction = KNN_Model.predict(X_test)
    y_prediction_train = KNN_Model.predict(X_train)

    return y_prediction, y_prediction_train, KNN_Model


####################################################### NN: Neural Network
def NN(X_train, X_test, y_train, y_test, user_params, verbose):

    # prepare dataset with input and output scalers, can be none
    def get_dataset(input_scaler, output_scaler):

        trainX, testX = X_train, X_test
        trainy, testy = y_train, y_test
        # scale inputs
        if input_scaler is not None:
            # fit scaler
            input_scaler.fit(trainX)
            # transform training dataset
            trainX = input_scaler.transform(trainX)
            # fit scaler
            # input_scaler.fit(testX)
            # transform test dataset
            testX = input_scaler.transform(testX)
        if output_scaler is not None:
            # reshape 1d arrays to 2d arrays
            trainy = trainy.reshape(len(trainy), 1)
            testy = testy.reshape(len(testy), 1)
            # fit scaler on training dataset
            output_scaler.fit(trainy)
            # transform training dataset
            trainy = output_scaler.transform(trainy)
            # fit scaler on testing dataset
            # output_scaler.fit(testy)
            # transform test dataset
            testy = output_scaler.transform(testy)
        return trainX, trainy, testX, testy

    def denormalize(main_data, normal_data, scaler):

        main_data = main_data.reshape(-1, 1)
        normal_data = normal_data.reshape(-1, 1)
        # scaleObject = StandardScaler()
        scaler.fit_transform(main_data)
        denormalizedData = scaler.inverse_transform(normal_data)

        return denormalizedData
    
    trainX, trainy, testX, testy = get_dataset(MinMaxScaler(), MinMaxScaler())
    
    # default parameters
    parameters = {'layers_neurons':[(trainX.shape[1]) // 2 + 1], 'activation':'exponential', 'loss':'mean_squared_error',
                  'optimizer':'RMSprop', 'metrics':['mean_squared_error'],
                  'EarlyStopping_monitor':'val_loss', 'EarlyStopping_patience':30, 'batch_size':128,
                  'validation_split':0.2,'epochs':100}
    
    if user_params is not None:
        for key in parameters.keys():
            if key in user_params.keys():
                parameters[key] = user_params[key]


    NeuralNetworkObject = keras.Sequential()
    NeuralNetworkObject.add(tf.keras.Input(shape=(trainX.shape[1],)))
    for layer,neurons in enumerate(parameters['layers_neurons']):
        NeuralNetworkObject.add(tf.keras.layers.Dense(neurons))
    NeuralNetworkObject.add(tf.keras.layers.Dense(1, activation=parameters['activation']))
    
    
    # Compile the model
    NeuralNetworkObject.compile(
        loss=parameters['loss'],
        optimizer=parameters['optimizer'],
        metrics=parameters['metrics'])

    early_stop = EarlyStopping(monitor=parameters['EarlyStopping_monitor'], patience=parameters['EarlyStopping_patience'])

    NeuralNetworkObject.fit(trainX, trainy.ravel(),
                   callbacks=[early_stop],
                   batch_size=parameters['batch_size'],
                   validation_split=parameters['validation_split'],
                   epochs=parameters['epochs'], verbose=0)
    
    if verbose == 2:
        test_mse = NeuralNetworkObject.evaluate(testX, testy)[1]
        print('NN mse test: ', test_mse)
        train_mse = NeuralNetworkObject.evaluate(trainX, trainy)[1]
        print('NN mse train: ', train_mse)
        
    y_prediction = NeuralNetworkObject.predict(testX)
    y_prediction = denormalize(y_train, y_prediction, MinMaxScaler())
    y_prediction_train = NeuralNetworkObject.predict(trainX)
    y_prediction_train = denormalize(y_train, y_prediction_train, MinMaxScaler())
    
    return np.array(y_prediction).ravel(), np.array(y_prediction_train).ravel(), NeuralNetworkObject