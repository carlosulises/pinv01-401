from keras.layers import Conv2D

def conv(layer, image_shape):
    #Se valida que la capa no sea nula
    if(layer is None):
        raise Exception("Error, la capa no se encuentra cargada.")
    
    #Se verifica que el valor no sea nulo
    if(layer["value"] is None):
        raise Exception("Error, las propiedades de la capa no se encuentra cargada.")
    else:
        layer_value = layer["value"]

    conveval = "Conv2D("

    #Se verifica la cantidad de filtros utilizados
    try:
        filters = layer_value["filters"]
        if(filters is None):
            raise Exception("Error, no posee el atributo requerido.")
    except KeyError:
        raise Exception("Error, no posee el atributo requerido.")

    conveval += "filters=%s, " % (str(filters))

    #Se verifica el tamaño de los filtros utilizados
    try:
        kernel_size = layer_value["kernel_size"]
        if(kernel_size is None):
            raise Exception("Error, no posee el atributo requerido.")
        
        if(kernel_size["width"] is None):
            raise Exception("Error, no posee el atributo de width requerido.")
        
        if(kernel_size["height"] is None):
            raise Exception("Error, no posee el atributo de height requerido.")
        
        kernel_size_value = (kernel_size["width"],kernel_size["height"])
    except KeyError:
        raise Exception("Error, no posee el atributo requerido.")

    conveval += "kernel_size=%s, " % (str(kernel_size_value))

    #Se verifica el tamaño de los pasos de los filtros utilizados
    try:
        pstrides = layer_value["strides"]
        if((pstrides is None) == False):
            if(pstrides["x"] is None):
                raise Exception("Error, no posee el atributo x del strides requerido.")
            
            if(pstrides["y"] is None):
                raise Exception("Error, no posee el atributo y del strides requerido.")
            
            strides_value = (pstrides["x"],pstrides["y"])

            conveval += "strides=%s, " % (str(strides_value))
    except KeyError:
        pstrides = None

    

    #Se verifica el relleno utilizado
    try:
        ppadding = layer_value["padding"]
        if((ppadding is None) == False):
            conveval += "padding='%s', " % (str(ppadding))
    except KeyError:
        ppadding = None

    #Se verifica el formato de salida de la convolucion utilizado
    try:
        pdata_format = layer_value["data_format"]
        if((pdata_format is None) == False):
            conveval += "data_format='%s', " % (str(pdata_format))
    except KeyError:
        pdata_format = None

    #Se verifica la dilatacion de la convolucion utilizada
    try:
        dilation_rate = layer_value["dilation_rate"]
        if(dilation_rate is None):
            raise Exception("Error, no posee el atributo requerido para su utilizacion.")
        
        if(dilation_rate["x"] is None):
            raise Exception("Error, no posee el atributo x requerido para su utilizacion.")
        
        if(dilation_rate["y"] is None):
            raise Exception("Error, no posee el atributo y requerido para su utilizacion.")
        
        dilation_rate_value = (dilation_rate["x"],dilation_rate["y"])
        conveval += "dilation_rate=%s, " % (str(dilation_rate_value))
    except Exception:
        dilation_rate_value = None

    #Se verifica la activation utilizada
    try:
        pactivation = layer_value["activation"]
        if((pactivation is None) == False):
            conveval += "activation='%s', " % (str(pactivation))
    except KeyError:
        pactivation = None

    #Se verifica la utilizacion use_bias se recomienda que siempre sea true
    try:
        puse_bias = layer_value["use_bias"]
        if((puse_bias is None) == False):
            conveval += "use_bias=%s, " % (str(puse_bias))
    except KeyError:
        puse_bias = None

    #Se verifica la utilizacion kernel_initializer forma en la que se inicializa los filtros
    try:
        pkernel_initializer = layer_value["kernel_initializer"]
        if((pkernel_initializer is None) == False):
            conveval += "kernel_initializer='%s', " % (str(pkernel_initializer))
    except KeyError:
        pkernel_initializer = None

    #Se verifica la utilizacion bias_initializer forma en la que se inicializa las bias
    try:
        pbias_initializer = layer_value["bias_initializer"]
        if((pbias_initializer is None) == False):
            conveval += "bias_initializer='%s', " % (str(pbias_initializer))
    except KeyError:
        pbias_initializer = None

    #Se verifica la utilizacion kernel_regularizer regularizacion para el kernel
    try:
        pkernel_regularizer = layer_value["kernel_regularizer"]
        if((pkernel_regularizer is None) == False):
            conveval += "kernel_regularizer='%s', " % (str(pkernel_regularizer))
    except KeyError:
        pkernel_regularizer = None

    #Se verifica la utilizacion bias_regularizer regularizacion para la bias
    try:
        pbias_regularizer = layer_value["bias_regularizer"]
        if((pbias_regularizer is None) == False):
            conveval += "bias_regularizer='%s', " % (str(pbias_regularizer))
    except KeyError:
        pbias_regularizer = None

    #Se verifica la utilizacion activity_regularizer regularizacion para la actividad
    try:
        pactivity_regularizer = layer_value["activity_regularizer"]
        if((pactivity_regularizer is None) == False):
            conveval += "activity_regularizer='%s', " % (str(pactivity_regularizer))
    except KeyError:
        pactivity_regularizer = None

    #Se verifica la utilizacion kernel_constraint constraint para el kernel
    try:
        pkernel_constraint = layer_value["kernel_constraint"]
        if((pkernel_constraint is None) == False):
            conveval += "kernel_constraint='%s', " % (str(pkernel_constraint))
    except KeyError:
        pkernel_constraint = None

    #Se verifica la utilizacion bias_constraint constraint para la bias
    try:
        pbias_constraint = layer_value["bias_constraint"]
        if((pbias_constraint is None) == False):
            conveval += "bias_constraint='%s', " % (str(pbias_constraint))
    except KeyError:
        pbias_constraint = None

    if(image_shape is not None):
        conveval += "input_shape=%s, " % (str(image_shape))


    conveval = conveval[:-2] + ")"
    print(conveval)
    #vareval = eval(conveval)
    #print(vareval)
    return eval(conveval)


