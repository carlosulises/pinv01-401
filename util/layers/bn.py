from keras.layers import BatchNormalization

def bn(layer):
    #Se valida que la capa no sea nula
    if(layer is None):
        raise Exception("Error, la capa no se encuentra cargada.")
    
    #Se verifica que el valor no sea nulo
    if(layer["value"] is None):
        raise Exception("Error, las propiedades de la capa no se encuentra cargada.")
    else:
        layer_value = layer["value"]

    bneval = "BatchNormalization(  "

    #Se verifica el atributo de axis a ser utilizado
    try:
        paxis = layer_value["axis"]
        if((paxis is None) == False):
            bneval += "axis=%s, " % (str(paxis))
    except KeyError:
        paxis = None

    #Se verifica el atributo de momentum a ser utilizado
    try:
        pmomentum = layer_value["momentum"]
        if((pmomentum is None) == False):
            bneval += "momentum=%s, " % (str(pmomentum))
    except KeyError:
        pmomentum = None

    #Se verifica el atributo de epsilon a ser utilizado
    try:
        pepsilon = layer_value["epsilon"]
        if((pepsilon is None) == False):
            bneval += "epsilon=%s, " % (str(pepsilon))
    except KeyError:
        pepsilon = None

    #Se verifica el atributo de center a ser utilizado
    try:
        pcenter = layer_value["center"]
        if((pcenter is None) == False):
            bneval += "center=%s, " % (str(pcenter))
    except KeyError:
        pcenter = None

    #Se verifica el atributo de scale a ser utilizado
    try:
        pscale = layer_value["scale"]
        if((pscale is None) == False):
            bneval += "scale=%s, " % (str(pscale))
    except KeyError:
        pscale = None

    #Se verifica el atributo de beta_initializer a ser utilizado
    try:
        pbeta_initializer = layer_value["beta_initializer"]
        if((pbeta_initializer is None) == False):
            bneval += "beta_initializer='%s', " % (str(pbeta_initializer))
    except KeyError:
        pbeta_initializer = None

    #Se verifica el atributo de gamma_initializer a ser utilizado
    try:
        pgamma_initializer = layer_value["gamma_initializer"]
        if((pgamma_initializer is None) == False):
            bneval += "gamma_initializer='%s', " % (str(pgamma_initializer))
    except KeyError:
        pgamma_initializer = None

    #Se verifica el atributo de moving_mean_initializer a ser utilizado
    try:
        pmoving_mean_initializer = layer_value["moving_mean_initializer"]
        if((pmoving_mean_initializer is None) == False):
            bneval += "moving_mean_initializer='%s', " % (str(pmoving_mean_initializer))
    except KeyError:
        pmoving_mean_initializer = None

    #Se verifica el atributo de moving_variance_initializer a ser utilizado
    try:
        pmoving_variance_initializer = layer_value["moving_variance_initializer"]
        if((pmoving_variance_initializer is None) == False):
            bneval += "moving_variance_initializer='%s', " % (str(pmoving_variance_initializer))
    except KeyError:
        pmoving_variance_initializer = None

    #Se verifica el atributo de beta_regularizer a ser utilizado
    try:
        pbeta_regularizer = layer_value["beta_regularizer"]
        if((pbeta_regularizer is None) == False):
            bneval += "beta_regularizer=%s, " % (str(pbeta_regularizer))
    except KeyError:
        pbeta_regularizer = None

    #Se verifica el atributo de gamma_regularizer a ser utilizado
    try:
        pgamma_regularizer = layer_value["gamma_regularizer"]
        if((pgamma_regularizer is None) == False):
            bneval += "gamma_regularizer=%s, " % (str(pgamma_regularizer))
    except KeyError:
        pgamma_regularizer = None

    #Se verifica el atributo de beta_constraint a ser utilizado
    try:
        pbeta_constraint = layer_value["beta_constraint"]
        if((pbeta_constraint is None) == False):
            bneval += "beta_constraint=%s, " % (str(pbeta_constraint))
    except KeyError:
        pbeta_constraint = None

    #Se verifica el atributo de gamma_constraint a ser utilizado
    try:
        pgamma_constraint = layer_value["gamma_constraint"]
        if((pgamma_constraint is None) == False):
            bneval += "gamma_constraint=%s, " % (str(pgamma_constraint))
    except KeyError:
        pgamma_constraint = None

    bneval = bneval[:-2] + ")"
    print(bneval)

    return eval(bneval)