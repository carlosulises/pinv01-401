from keras.layers import MaxPooling2D

def max_pool(layer):
    #Se valida que la capa no sea nula
    if(layer is None):
        raise Exception("Error, la capa no se encuentra cargada.")
    
    #Se verifica que el valor no sea nulo
    if(layer["value"] is None):
        raise Exception("Error, las propiedades de la capa no se encuentra cargada.")
    else:
        layer_value = layer["value"]

    #Se verifica el tamaño de los pasos de los filtros utilizados
    try:
        pool_size = layer_value["pool_size"]
        if(pool_size is None):
            raise Exception("Error, no posee el atributo requerido.")
        
        if(pool_size["vertical"] is None):
            raise Exception("Error, no posee el atributo vertical del strides requerido.")
        
        if(pool_size["horizontal"] is None):
            raise Exception("Error, no posee el atributo horizontal del strides requerido.")
        
        pool_size_value = (pool_size["vertical"],pool_size["horizontal"])
    except KeyError:
        raise Exception("Error, no posee el atributo requerido.")

    maxpooleval = "MaxPooling2D(pool_size=%s, " % (str(pool_size_value))

     #Se verifica el tamaño de los pasos de los filtros utilizados
    try:
        pstrides = layer_value["strides"]
        if((pstrides is None) == False):
            if(pstrides["x"] is None):
                raise Exception("Error, no posee el atributo x del strides requerido.")
            
            if(pstrides["y"] is None):
                raise Exception("Error, no posee el atributo y del strides requerido.")
            
            strides_value = (pstrides["x"],pstrides["y"])

            maxpooleval += "strides=%s, " % (str(strides_value))
    except KeyError:
        pstrides = None

    #Se verifica el tamaño del relleno utilizado
    try:
        ppadding = layer_value["padding"]
        if((ppadding is None) == False):
            maxpooleval += "padding='%s', " % (str(ppadding))
    except KeyError:
        ppadding = None

    #Se verifica el formato de salida de la convolucion utilizado
    try:
        pdata_format = layer_value["data_format"]
        if((pdata_format is None) == False):
            maxpooleval += "data_format='%s', " % (str(pdata_format))
    except KeyError:
        pdata_format = None

    maxpooleval = maxpooleval[:-2] + ")"
    print(maxpooleval)

    return eval(maxpooleval)