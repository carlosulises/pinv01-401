import sys
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import gc

def construct_dnn(exe_amount, properties=None, wsave=None, shape=None, X_train=None, Y_train=None):
    # The architecture defined in json format is loaded
    layers = properties.dnn()
    # Loads allowed layers
    architecture = properties.model_architecture()

    if(layers is None):
        raise Exception("Error, the application model is not loaded.")

    if(architecture is None):
        raise Exception("Error, the application architecture is not loaded.")
    
    # The Keras Sequential class is instantiated
    model = Sequential()
    cont = 0
    # The path is loaded with the allowed layers that are going to be passed to the model dynamically
    sys.path.append('./util/layers')
    # The charged layers are traversed
    for layer in layers:
        try:
            # The model is loaded dynamically depending on the type
            func = str(architecture[cont])
            module = import_from(func)
            if(func == "conv" or func == "fc"):
                if(cont != 0):
                    shape = None
                model.add(eval('module.%s(layer, shape)' % (func)))
            else:
                model.add(eval('module.%s(layer)' % (func)))
            cont = cont + 1
        except:
            raise Exception("Error, the model could not be generated dynamically.")
    if properties.weights() == "same":
        if not (wsave):
            print("The weights of the first execution are saved.")
            wsave=model.get_weights()
        else:
            print("The weights are loaded for execution.")
            model.set_weights(wsave)
    
    # The structure of the architecture is printed
    print(model.summary())
    # Interval controller added
    if(properties.metric_enable() == True or properties.analyze()):
        weight_save_callback = ModelCheckpoint('data/check/weights_'+exe_amount+'.{epoch:02d}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, mode='auto', period=properties.interval())
    else :
        if(properties.layering_optimization() == True):
            weight_save_callback = ModelCheckpoint('data/check/weights_'+exe_amount+'.hdf5', monitor='val_loss', verbose=0, save_best_only=False, mode='auto')
    # The model is compiled
    model.compile(loss=properties.loss(), optimizer=properties.optimizer(), metrics=['accuracy'])
    # The model is trained
    model.fit(X_train, Y_train, epochs=properties.epochs(), batch_size=properties.batch_size(), verbose=2,callbacks=[weight_save_callback])
    # The model is generated
    model.save('data/check/my_model_'+exe_amount+'.h5')

    return model, wsave

def import_from(s):
    module = __import__(s)
    return module