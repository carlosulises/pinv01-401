from keras.layers import Dense
from keras import regularizers

def fc(layer, image_shape):
    #Se valida que la capa no sea nula
    if(layer is None):
        raise Exception("Error, la capa no se encuentra cargada.")
    
    #Se verifica que el valor no sea nulo
    if(layer["value"] is None):
        raise Exception("Error, las propiedades de la capa no se encuentra cargada.")
    else:
        layer_value = layer["value"]

    #Se verifica la cantidad de neuronas utilizadas
    try:
        units = layer_value["units"]
        if(units is None):
            raise Exception("Error, no posee el atributo units requerido.")
    except KeyError:
        raise Exception("Error, no posee el atributo units requerido.")

    denseeval = "Dense(%s, " % (str(units))

    #Se verifica el tamaño de la entrada a la primera capa
    '''try:
        pinput_shape = layer_value["input_shape"]
        if((pinput_shape is None) == False):
            denseeval += "input_shape=%s, " % (str(pdata_format))
    except KeyError:
        pinput_shape = None'''

    #Se verifica el tamaño de la entrada a la primera capa
    try:
        pactivation = layer_value["activation"]
        if((pactivation is None) == False):
            denseeval += "activation='%s', " % (str(pactivation))
    except KeyError:
        pactivation = None

    #Se verifica el la utilizacion del bias en las capas
    try:
        puse_bias = layer_value["use_bias"]
        if((puse_bias is None) == False):
            denseeval += "use_bias=%s, " % (str(puse_bias))
    except KeyError:
        puse_bias = None

    #Se verifica la inicializacion del kernel en las capas
    try:
        pkernel_initializer = layer_value["kernel_initializer"]
        if((pkernel_initializer is None) == False):
            denseeval += "kernel_initializer='%s', " % (str(pkernel_initializer))
    except KeyError:
        pkernel_initializer = None

    #Se verifica la inicializacion del bias en las capas
    try:
        pbias_initializer = layer_value["bias_initializer"]
        if((pbias_initializer is None) == False):
            denseeval += "bias_initializer='%s', " % (str(pbias_initializer))
    except KeyError:
        pbias_initializer = None

    #Se verifica la regularizacion del kernel en las capas
    try:
        pkernel_regularizer = layer_value["kernel_regularizer"]
        if((pkernel_regularizer is None) == False):
            denseeval += "kernel_regularizer='%s', " % (str(pkernel_regularizer))
    except KeyError:
        pkernel_regularizer = None

    #Se verifica la regularizacion del bias en las capas
    try:
        pbias_regularizer = layer_value["bias_regularizer"]
        if((pbias_regularizer is None) == False):
            denseeval += "bias_regularizer= %s, " % (str(pbias_regularizer))
    except KeyError:
        pbias_regularizer = None
    
    #Se verifica la regularizacion del activity en las capas
    try:
        pactivity_regularizer = layer_value["activity_regularizer"]
        if((pactivity_regularizer is None) == False):
            denseeval += "activity_regularizer='%s', " % (str(pactivity_regularizer))
    except KeyError:
        pactivity_regularizer = None

    #Se verifica la constraint del kernel en las capas
    try:
        pkernel_constraint = layer_value["kernel_constraint"]
        if((pkernel_constraint is None) == False):
            denseeval += "kernel_constraint='%s', " % (str(pkernel_constraint))
    except KeyError:
        pkernel_constraint = None

    #Se verifica la constraint del bias en las capas
    try:
        pbias_constraint = layer_value["bias_constraint"]
        if((pbias_constraint is None) == False):
            denseeval += "bias_constraint=%s, " % (str(pbias_constraint))
    except KeyError:
        pbias_constraint = None
    
    if(image_shape is not None):
        denseeval += "input_shape=%s, " % (str(image_shape))

    denseeval = denseeval[:-2] + ")"
    print(denseeval)

    return eval(denseeval)