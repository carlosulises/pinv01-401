from keras.layers import Flatten

def flat(layer):

    #Se valida que la capa no sea nula
    if(layer is None):
        raise Exception("Error, la capa no se encuentra cargada.")
    
    #Se verifica que el valor no sea nulo
    if(layer["value"] is None):
        flateval = "Flatten()"
    else:
        layer_value = layer["value"]
        #Se verifica el formato de salida de la convolucion utilizado
        try:
            pdata_format = layer_value["data_format"]
            if((pdata_format is None) == False):
                flateval = "Flatten(data_format=%s)" % (str(pdata_format))
            else:
                flateval ="Flatten()"
        except KeyError:
            pdata_format = None
            flateval ="Flatten()"
            
    print(flateval)
    return eval(flateval)
