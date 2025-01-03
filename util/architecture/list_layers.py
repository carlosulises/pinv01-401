import os

def list_layers():
    # Abrir un archivo
    path = "./util/layers/"
    dirs = os.listdir( path )

    return dirs

#print(list_layers())