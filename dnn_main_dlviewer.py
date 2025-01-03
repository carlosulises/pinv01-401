
from util.architecture.dl_config import *
from util.pretreatment.kt_utils import *
from util.pretreatment.normalize import *
from util.construct_dnn import *
from time import time
from util.optimization import optimization, optimization_analyze

def dnn_main():
    # The class responsible for obtaining the properties is instantiated.
    properties = ConfigParseLD("config.ini")
    wsave=None
    # If multiple executions are configured
    for i in (np.arange(int(properties.amount()))):
        print('::::::::::: Execution Num: --- ' + str(i+1)+' :::::::::::')
        # How to use the dataset
        mode = properties.mode()
        print("Execution mode: "+ mode)
        # Loading datasets
        X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset(properties.databaset_type(), mode, properties.databaset(), properties.train(), properties.test(), properties.archives_attributes())
        print("Training and testing dataset loaded....")

        # The dataset is normalized and the size is verified
        X_train, X_test, Y_train, Y_test, shape = normalize_dataset(properties.databaset_type(), X_train_orig, Y_train_orig, X_test_orig, Y_test_orig)
        
        # Start time counting.
        start_time = time()
        # The model is generated, compiled and trained in the traditional way
        model, wsave = construct_dnn(str(i), properties, wsave, shape, X_train, Y_train)
        # Calculate the elapsed time.
        elapsed_time = time() - start_time
        print('Elapsed time : ', str(str(elapsed_time)+' seconds'))

        print("Layer optimization")
        optimization_analyze(str(i), mode, properties, Y_train, Y_test, X_train, X_test)









