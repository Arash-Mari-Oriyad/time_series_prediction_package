def train_evaluate(training_data, validation_data, model, model_parameters = None, verbose = 1):
    
    supported_models_name = ['gbm', 'glm', 'knn', 'nn']
    train_predictions = None
    validation_predictions = None
    trained_model = None
    
    if type(training_data) == str:
        training_data = pd.read_csv(training_data)
    if type(validation_data) == str:
        validation_data = pd.read_csv(validation_data)
        
    # split features and target
    X_training = training_data.drop(['Target','spatial id','temporal id'],axis = 1)
    Y_training = np.array(training_data['Target']).reshape(-1)
    X_validation = validation_data.drop(['Target','spatial id','temporal id'],axis = 1)
    Y_validation = np.array(validation_data['Target']).reshape(-1)
    
        
    if (type(model) == str) and (model in supported_models_name) :
        
        if model == 'gbm':
            validation_predictions, train_predictions, trained_model = GBM(X_training, X_validation, Y_training, model_parameters, verbose)
        elif model == 'glm':
            validation_predictions, train_predictions, trained_model = GLM(X_training, X_validation, Y_training, model_parameters, verbose)
        elif model == 'knn':
            validation_predictions, train_predictions, trained_model = KNN(X_training, X_validation, Y_training, model_parameters, verbose)
        elif model == 'nn':
            validation_predictions, train_predictions, trained_model = NN(X_training, X_validation, Y_training, Y_validation, model_parameters, verbose)
        
    elif callable(model):
        try:
            validation_predictions, train_predictions, trained_model = model(X_training, X_validation, Y_training)
        except ValueError:
            sys.exit("The user-defined function is not compatible with the definition.")
        
    else:
        sys.exit("The model must be name of one of the supported models: {'gbm', 'glm', 'knn', 'nn'}\nOr user defined function.")
        
    return train_predictions, validation_predictions, trained_model
    