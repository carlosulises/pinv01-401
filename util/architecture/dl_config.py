import configparser
import json
import os
from util.architecture.dnn import *
from util.architecture.list_layers import *

# Clase utilizada para la lectura de las propiedades del archivo de configuracion
class ConfigParseLD():
    # Constructor utilizado para la lectura del archivo de propiedades
    def __init__(self,file):
        try:
            print('Archivo de configuracion: '+file)
            self.config = configparser.ConfigParser()
            self.config.read(file)

            modelconfigpath = self.config['net']['dnn']
            layers = self.config['net']['layers']
            layerings_conf = self.config['net']['layerings']
            self.layerings = check_dnn_layerings_architecture(layerings_conf)

            print("Ruta del modelo: "+ modelconfigpath)

            self.model_architecture_supported = list_layers()

            if os.path.isfile(modelconfigpath):
                with open(modelconfigpath) as config_file:
                    self.model = json.load(config_file)
                    self.architecture = check_dnn_architecture(self.model, layers, self.model_architecture_supported)
            else:
                raise Exception("Error, el archivo de configuracion del modelo de red neuronal produnfa no existe en la ruta indicada.")
        except:
            raise Exception("Error, al intentar parsear la configuracion de la red.")
    
    # Metodos que devuelven las distintas propiedades utilizadas
    ##########  optimizador  ##########
    def optimizer(self):
        return self.model['optimizer']['optimizer']
    def loss(self):
        return self.model['optimizer']['loss']
    ##########  archives  ##########
    def databaset_type(self):
        return self.model['archives']['type']
    def databaset(self):
        return self.model['archives']['dataset']
    def train(self):
        return self.model['archives']['dataset_train']
    def test(self):
        return self.model['archives']['dataset_test']
    def mode(self):
        return self.model['archives']['mode']
    ##########  red  ##########
    def batch_size(self):
        return self.model['execution']['batch_size']
    def epochs(self):
        return self.model['execution']['epochs']
    def interval(self):
        return self.model['execution']['interval']
    def amount(self):
        return self.model['execution']['amount']
    def method(self):
        return self.model['execution']['method']
    def method_metric(self):
        if self.model['execution']['method'] is None:
            raise Exception("Error, el parámetro de ejecución no asignado .")
        return self.model['execution']['method']
    def weights(self):
        return self.model['execution']['weights']
    def dnn_model(self):
        return  self.model
    def dnn(self):
        return  self.model["dnn"]
    def dnn_size(self):
        return  len(self.model["dnn"])
    def model_architecture(self):
        return  self.architecture

    def metric_type(self):
        return self.model['execution']['catch_type']

    def archives_attributes(self):
        return self.model['archives']['attributes']

    ##########  metric  ##########

    def metrics(self):
        return self.model['metric']


    def result_type(self):
        return self.model['result']['type']

    def result_summary(self):
        return self.model['result']['summary']

    def result_ylim_start(self):
        if self.model['result']['ylim'] is None:
            raise Exception("Error, el parámetro de resumen para los graficos start no se encuentra asignado ")
        return self.model['result']['ylim']['start']

    def result_ylim_end(self):
        if self.model['result']['ylim'] is None:
            raise Exception("Error, el parámetro de resumen para los graficos end no se encuentra asignado ")
        return self.model['result']['ylim']['end']

    def layering_optimization(self):
        return self.model['layeringOptimization']
    def optimization_reset_weight_bias(self):
        return self.model['resetWeightBias']

    def metric_enable(self):
        return self.model['metricEnable']
    def analyze(self):
        return self.model['analyze']
    def max_thread(self):
        return self.model['metric']['max_thread']
    def lot(self):
        return self.model['metric']['lot']
    def perceptron(self):
        return self.model['metric']['perceptron']
    def module(self):
        return self.model['metric']['module']
    def parameters(self):
        return self.model['metric']['parameters']
    
    def epoch_measure(self):
        return self.model['metric']['epoch_measure']
    def lot_measure(self):
        return self.model['metric']['lot_measure']
    def metric_type_array(self):
        return self.layerings
