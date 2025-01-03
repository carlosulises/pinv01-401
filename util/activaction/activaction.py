from keras import backend as K
from keras.models import Model
from keras.layers import Input, Flatten
from sklearn import preprocessing
import gc

def get_activaction_k(model, input_layer, output_layer, x):
    if(model is None):
        raise Exception("Error, el modelo se encuentra vacio.")

    if(input_layer is None):
        raise Exception("Error, la capa de entrada se encuenta vacia.")

    if(output_layer is None):
        raise Exception("Error, la capa de salida se encuenta vacia.")

    if(x is None):
        raise Exception("Error, los datos se encuentran vacios.")

    get_l_output = K.function([model.layers[input_layer].input],
                                    [model.layers[output_layer].output])
    l_output = get_l_output([x])[0]

    return l_output

def get_activaction_model_optimizate(model, layer_name, data, batch_size):
    if(model is None):
        raise Exception("Error, el modelo se encuentra vacio.")

    if(layer_name is None):
        raise Exception("Error, el nombre de la capa es vacia.")
    
    print("layer_name: "+ str(layer_name))

    intermediate_layer_model = Model(inputs=model.input,
                                    outputs=model.layers[layer_name].get_output_at(-1))
    intermediate_output = intermediate_layer_model.predict(data, batch_size=batch_size)
    del intermediate_layer_model
    gc.collect()
    return intermediate_output

def get_activaction_Model(model, layer_name, data):
    if(model is None):
        raise Exception("Error, el modelo se encuentra vacio.")

    if(layer_name is None):
        raise Exception("Error, el nombre de la capa es vacia.")
    
    print("layer_name: "+ str(layer_name))

    intermediate_layer_model = Model(inputs=model.input,
                                    outputs=model.layers[layer_name].get_output_at(-1))
    intermediate_output = intermediate_layer_model.predict(data, batch_size=0)
    del intermediate_layer_model
    gc.collect()

    '''inShape = intermediate_output.shape
    if len(inShape) == 2:
        inputs = Input(shape=(inShape[1],))
    else:
        if len(inShape) == 3:
            inputs = Input(shape=(inShape[1],inShape[2],1))
        else:
            inputs = Input(shape=(inShape[1],inShape[2],inShape[3]))

    prediction = Flatten()(inputs)
    model = Model(inputs=inputs, outputs=prediction)
    
    print("Medida del ouput ---->")
    print(intermediate_output.shape)
    print(len(intermediate_output.shape))
    X = model.predict(intermediate_output)

    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit(X).transform(X)

    print("Medida del ouput ---->")
    print(X.shape)
    print(len(X.shape))'''

    return intermediate_output

def get_activaction(input_dim, weights):
    if(input_dim is None):
        raise Exception("Error, el tamaño de entrada se encuentra vacio.")

    if(weights is None):
        raise Exception("Error, los pesos de entrada se encuenta vacia.")

    model = Sequential()
    model.add(Dense(int(cant_nodos), input_dim=int(cant_input), weights=weights, activation=activation))
    activations = model.predict_proba(activations)
    activations_array = np.asarray(activations)
    return activations_array