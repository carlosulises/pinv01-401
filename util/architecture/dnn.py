
def check_dnn_architecture(architecture=None, layer=None, supported= None):
    if(architecture is None):
        raise Exception("Error, no se cargo el modelo.")
    if(layer is None):
        raise Exception("Error, no se cargo las capas utilizadas.")
    if(supported is None):
        raise Exception("Error, no se enecuntra cargado ningún tipo de capa.")
    
    layers = layer.split(",")
    archit = architecture["dnn"]

    if(archit is None):
        raise Exception("Error, no se cargo la arquitectura del modelo.")
    
    use_layer = []
    cont = 0
    for temp in archit:
        try:
            layer_type = temp["type"]
            if(layer_type is None):
                raise Exception("Error, la capa Nro.: "+ str(cont)+" no posee el atributo type.")
        except KeyError:
            raise Exception("Error, la capa Nro.: "+ str(cont)+" no posee el atributo type.")
        
        if layer_type in layers:
            if(layer_type+".py" in supported):
                use_layer.append(layer_type)
            else:
                raise Exception("Error, la capa Nro.: "+ str(cont)+" no posee una capa aceptable por el sistema de capas cargados.")
        else:
            raise Exception("Error, la capa Nro.: "+ str(cont)+" no posee una capa no manejable por el sistema, verifique la configuración de la aplicación.")
        
        cont = cont + 1
    return use_layer


def check_dnn_layerings_architecture(layer=None):
    if(layer is None):
        raise Exception("Error, no se cargo las capas del layerings.")

    layers = layer.split(",")

    return layers