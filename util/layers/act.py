from keras.layers import Activation

def act(layer):
    #Se valida que la capa no sea nula
    if(layer is None):
        raise Exception("Error, la capa no se encuentra cargada.")
    
    #Se verifica que el valor no sea nulo
    if(layer["value"] is None):
        raise Exception("Error, las propiedades de la capa no se encuentra cargada.")
    else:
        layer_value = layer["value"]

    #Se verifica el relleno utilizado
    try:
        pfunction = layer_value["function"]
        if(pfunction is None):
            raise Exception("Error, no posee el atributo function requerido.")
    except KeyError:
        raise Exception("Error, no posee el atributo function requerido.")

    funceval = "Activation('%s')" % (str(pfunction))
    print(funceval)
    
    return eval(funceval)