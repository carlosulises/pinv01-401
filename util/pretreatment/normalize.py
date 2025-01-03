
def normalize_dataset(typeDataset = None,X_train_orig = None, Y_train_orig = None, X_test_orig = None, Y_test_orig = None):
    # feature variable
    if X_train_orig is None:
        raise Exception("Error, feature train dataset is null")

    if X_test_orig is None:
        raise Exception("Error, test feature dataset is null.")
    
    # results variable
    if Y_train_orig is None:
        raise Exception("Error, the results train dataset is null.")

    if Y_test_orig is None:
        raise Exception("Error, test result dataset is null.")
    # Depending on the mode, it is normalized and the size is changed.
    if typeDataset == "h5":
        if len(X_train_orig.shape) == 3:
            X_train_orig = X_train_orig.reshape(X_train_orig.shape[0], X_train_orig.shape[1], X_train_orig.shape[2], 1)
            X_test_orig = X_test_orig.reshape(X_test_orig.shape[0], X_test_orig.shape[1], X_test_orig.shape[2], 1)
        
        print("The image vectors are normalized...")
        # Normalize image vectors
        X_train = X_train_orig/255.
        X_test = X_test_orig/255.

        # Reshape
        Y_train = Y_train_orig.T
        Y_test = Y_test_orig.T
        print("The matrix is ​​reformed...")

        if len(X_test.shape) == 2:
            shape = (X_test.shape[1],)
        else:
            if len(X_test.shape) == 3:
                shape = (X_test.shape[1], X_test.shape[2],1)
            else:
                shape = (X_test.shape[1],X_test.shape[2],X_test.shape[3])

    if typeDataset == "csv":
        X_train = X_train_orig
        X_test = X_test_orig

        Y_train = Y_train_orig.T
        Y_test = Y_test_orig.T
        
        shape = (X_test.shape[1],)

    print ("Number of training examples = " + str(X_train.shape[0]))
    print ("Number of testing examples = " + str(X_test.shape[0]))
    print ("X_train size: " + str(X_train.shape))
    print ("Y_train size: " + str(Y_train.shape))
    print ("X_test size: " + str(X_test.shape))
    print ("Y_test size: " + str(Y_test.shape))
    print ("input size: " + str(shape))

    return X_train, X_test, Y_train, Y_test, shape