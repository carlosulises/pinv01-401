from functools import partial
from numbers import Real
from keras.models import load_model
import tensorflow.python.keras.backend as K
from keras.models import Model
from keras.models import Sequential
from util.activaction.activaction import get_activaction_model_optimizate
from util.construct_dnn import *
from keras.layers import Dense
from keras import regularizers
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
from time import time
import gc
from skopt import BayesSearchCV
from keras.optimizers import Adam, SGD, RMSprop
from keras.wrappers.scikit_learn import KerasClassifier

shared_var = 0

def get_layer(model):
    cont = 0
    return_layer = None
    for layer in model.layers:
        if cont == 0:
            return_layer = layer
            break
    return return_layer

def get_last_layer(model):
    cont = 0
    return_layer = None
    for layer in model.layers:
        if cont == 1:
            return_layer = layer
            break
        cont = cont +1
    return return_layer

# Defines the function to build the model
def create_model(optimizer='adam', learning_rate=0.001):
    print("-----------------------------------------------------------")
    print("The model is loaded: "+ str(shared_var))
    model = get_model ('my_model_0', shared_var)
    
    # Create the optimizer with the learning rate
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    
    # Compile the model
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def optimization_analyze(exe_amount, mode, properties, Y_train, Y_test, X_train, X_test,):
    K.clear_session()
    gc.collect()

    # The datasets are concatenated
    inputs = np.concatenate((X_train, X_test))
    targets = np.concatenate((Y_train, Y_test))

    # The intervals and epoch are obtained
    inter = range(properties.interval(),(properties.epochs()+properties.interval()),properties.interval())
    epochs = [{} for x in inter]
    c = len(epochs)
    
    # Defines the search space for the hyperparameters
    param_space = {
        'optimizer': ['adam', 'sgd', 'rmsprop'],  # Optimizer
        'learning_rate': (1e-4, 1e-2, 'log-uniform'),  # Learning rate
        'batch_size': (16, 64)  # Rango de batch_size
    }

    # The global variable is defined
    global shared_var
    shared_var = 2
    
    # Classification model
    model = KerasClassifier(
        build_fn=create_model,
        epochs=properties.epochs(),
        batch_size=properties.batch_size(),
        verbose=2  
    )

    # Perform Bayesian Optimization
    bayes_search = BayesSearchCV(
        estimator=model,
        search_spaces=param_space,
        cv=10,  # Cross validation
        n_iter=properties.epochs(),  # Number of search iterations
        random_state=42
    )

    # Refine your search
    bayes_search_result = bayes_search.fit(inputs, targets)

    batch_size = bayes_search_result.best_params_['batch_size']
    learning_rate = bayes_search_result.best_params_['learning_rate']
    optimizer = bayes_search_result.best_params_['optimizer']

    print("Better hyperparameters:")
    print(f"Learning rate: {learning_rate}")
    print(f"Optimizer: {optimizer}")
    print(f"Batch No. per training: {batch_size}")

    acc_epoch_ori = []
    acc_epoch_opt = []
    time_epoch_ori = []
    time_epoch_opt = []

    for j in range(c):
        # The shared variable is changed to obtain the model corresponding to the epoch
        shared_var = inter[j]
        # Define the K-fold Cross Validator
        kfoldOri = KFold(n_splits=10, shuffle=True)
        # K-fold Cross Validation model evaluation
        acc_fold_ori = []
        time_ori = []
        # The validations are reviewed to build the model
        for train, test in kfoldOri.split(inputs, targets):            
            start_time = time()
            # The model is created
            modelsx = create_model(optimizer,learning_rate)
            # The model is trained
            modelsx.fit(inputs[train], targets[train], epochs=properties.epochs(), batch_size=batch_size, verbose=1)
            # The model is evaluated
            loss, mae = modelsx.evaluate(inputs[test], targets[test], verbose=1)
            elapsed_time = time() - start_time

            acc_tmp = mae*100
            acc_fold_ori.append(acc_tmp)
            time_ori.append(elapsed_time)
            # instance is deleted
            del modelsx
            gc.collect()
        # Se obtiene el modelo original
        original_model = create_model(optimizer,learning_rate)

        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=10, shuffle=True)
        # K-fold Cross Validation model evaluation
        acc_per_fold = []
        loss_per_fold = []
        time_epoch = []
        fold_no = 1
        # The validations are reviewed to build the model
        for train, test in kfold.split(inputs, targets):
            cont = 0
            last = len(original_model.layers) -1
            activaction_train = None
            activaction_test = None
            activaction_train_temp = None
            activaction_test_temp = None
            # the optimized model is generated
            optimization_model = Sequential()
            # Start time counting.
            start_time = time()
            # The layers are obtained
            for layer in original_model.layers:
                # Check if it is the last layer
                if cont < last:
                    # It is compared if the layer has zero parameters
                    if layer.count_params() == 0:
                        # The layer is added
                        optimization_model.add(layer)
                        if layer.__class__.__name__ == 'Flatten':
                            flatten_layer = 1
                            if original_model.layers[1].__class__.__name__ != 'Flatten':
                                flatten_layer = 2
                            # Activation is obtained with the optimized hyperparameters
                            activaction_train = get_activaction_model_optimizate(layer_model,  flatten_layer, activaction_train_temp, batch_size)
                            activaction_test = get_activaction_model_optimizate(layer_model, flatten_layer, activaction_test_temp, batch_size)
                    else:
                        # The weights and bias are initialized
                        weights_initializer = layer.kernel_initializer
                        bias_initializer = layer.bias_initializer
                        layer_cont = None
                        if original_model.layers[cont+1].count_params() == 0:
                            layer_cont = 1
                        else: 
                            layer_cont = 0
                        # The one-layer model is generated and trained with the dataset
                        layer_model = Sequential()
                        if cont == 0:
                            # weights and bias are reset
                            layer.set_weights([
                                weights_initializer(layer.kernel.shape, dtype=layer.kernel.dtype).numpy(),
                                bias_initializer(layer.bias.shape, dtype=layer.bias.dtype).numpy()
                            ])
                            layer_model.add(layer)
                            if original_model.layers[cont+1].count_params() == 0:
                                layer_model.add(original_model.layers[cont+1])
                            if layer.__class__.__name__ != 'Dense' and layer.__class__.__name__ != 'Flatten':
                                layer_model.add(Flatten())
                            layer_model.add(Dense(1, activation='sigmoid'))
                            layer_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
                            layer_model.fit(inputs[train], targets[train], epochs=properties.epochs(), batch_size=batch_size, verbose=2)
                            print(layer_model.summary())
                            # activation is obtained
                            activaction_train = get_activaction_model_optimizate(layer_model, layer_cont, inputs[train], batch_size)
                            activaction_test = get_activaction_model_optimizate(layer_model, layer_cont, inputs[test], batch_size)
                            # The layer is added to the optimized model
                            optimization_model.add(get_layer(layer_model))
                        if cont > 0:
                            # The one-layer model is generated and trained with the activation
                            # weights and bias are reset
                            layer.set_weights([
                                weights_initializer(layer.kernel.shape, dtype=layer.kernel.dtype).numpy(),
                                bias_initializer(layer.bias.shape, dtype=layer.bias.dtype).numpy()
                            ])

                            layer_model.add(layer)
                            if original_model.layers[cont+1].count_params() == 0:
                                layer_model.add(original_model.layers[cont+1])
                            if layer.__class__.__name__ != 'Dense' and layer.__class__.__name__ != 'Flatten':
                                layer_model.add(Flatten())
                            layer_model.add(Dense(1, activation='sigmoid'))
                            layer_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
                            layer_model.fit(activaction_train, targets[train], epochs=properties.epochs(), batch_size=batch_size, verbose=2)
                            print(layer_model.summary())
                            # activation is obtained
                            activaction_train_temp = activaction_train
                            activaction_test_temp = activaction_test
                            activaction_train = get_activaction_model_optimizate(layer_model,  layer_cont, activaction_train, batch_size)
                            activaction_test = get_activaction_model_optimizate(layer_model, layer_cont, activaction_test, batch_size)
                            # The layer is added to the optimized model
                            optimization_model.add(get_layer(layer_model))
                            if cont == (last -1):
                                optimization_model.add(get_last_layer(layer_model))
                cont = cont + 1
            # Calculate the elapsed time.
            elapsed_time = time() - start_time
            time_epoch.append(elapsed_time)
            # The optimized model is built
            optimization_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            optimization_model.build(inputs[train].shape)
            optimization_model.summary()
            
            # Generate generalization metrics
            scores = optimization_model.evaluate(inputs[test], targets[test], verbose=0, batch_size=batch_size)
            print(f'Score for fold {fold_no}: {optimization_model.metrics_names[0]} of {scores[0]}; {optimization_model.metrics_names[1]} of {scores[1]*100}%')
            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])

            # Increase fold number
            fold_no = fold_no + 1
            final_time = "%0.10f" % elapsed_time
            print('Elapsed time Optimization: ', str(str(final_time)+' seconds'))
        # == Provide average scores ==
        print('------------------------------------------------------------------------')
        print('Score per fold')
        for i in range(0, len(acc_per_fold)):
            print('------------------------------------------------------------------------')
            print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)}')
        print('------------------------------------------------------------------------')
        print('Original per fold')
        for i in range(0, len(acc_fold_ori)):
            print('------------------------------------------------------------------------')
            print(f'> Fold {i+1} - Accuracy: {acc_fold_ori[i]}%')
        print('------------------------------------------------------------------------')
        print('Average scores for all Original folds:')
        print(f'> Accuracy: {np.mean(acc_fold_ori)} (+- {np.std(acc_fold_ori)})')
        acc_epoch_ori.append(np.mean(acc_fold_ori))
        acc_epoch_opt.append(np.mean(acc_per_fold))

        print('------------------------------------------------------------------------')
        print('Optimization per time')
        for i in range(0, len(time_epoch)):
            print('------------------------------------------------------------------------')
            print(f'> Fold {i+1} - Time: {time_epoch[i]} seconds')
        print('------------------------------------------------------------------------')
        print('Average time for all Optimization folds:')
        print(f'> Time: {np.mean(time_epoch)} (+- {np.std(time_epoch)})')
        time_epoch_opt.append(np.mean(time_epoch))

        print('------------------------------------------------------------------------')
        print('Original per time')
        for i in range(0, len(time_ori)):
            print('------------------------------------------------------------------------')
            print(f'> Fold {i+1} - Time: {time_ori[i]} seconds')
        print('------------------------------------------------------------------------')
        print('Average time for all Original folds:')
        print(f'> Time: {np.mean(time_ori)} (+- {np.std(time_ori)})')
        time_epoch_ori.append(np.mean(time_ori))

    print('------------------------------------------------------------------------')
    print('Results obtained by epoch')
    print('Traditional:')
    print(acc_epoch_ori)
    print(time_epoch_ori)
    final_time_ori = "%0.10f" % np.mean(time_epoch_ori)
    print('Elapsed time Original: ', str(str(final_time_ori)+' seconds'))

    print('Optimization:')
    print(acc_epoch_opt)
    print(time_epoch_opt)
    final_time_opt = "%0.10f" % np.mean(time_epoch_opt)
    print('Elapsed time Optimization: ', str(str(final_time_opt)+' seconds'))
