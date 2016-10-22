from cnn import myCNN
import theano
import theano.tensor as T
from theano import shared
from theano import function
import numpy as np
import six.moves.cPickle as pickle
import gzip


def load_mnist_dataset(dataset_dir, dataset_filename):
    dataset_dir = dataset_dir + dataset_filename

    print "Loading the dataset"
    with gzip.open(dataset_dir, 'rb') as f:
        (trainSet, validSet, testSet) = pickle.load(f)

    (trainData, trainLabels) = trainSet
    (validData, validLabels) = validSet
    (testData, testLabels) = testSet

    shared_traindata = shared(np.asarray(trainData, dtype=theano.config.floatX), borrow=True)
    shared_trainlabels = shared(np.asarray(trainLabels.flatten(), dtype=theano.config.floatX), borrow=True)
    shared_validdata = shared(np.asarray(validData, dtype=theano.config.floatX), borrow=True)
    shared_validlabels = shared(np.asarray(validLabels.flatten(), dtype=theano.config.floatX), borrow=True)
    shared_testdata = shared(np.asarray(testData, dtype=theano.config.floatX), borrow=True)
    shared_testlabels = shared(np.asarray(testLabels.flatten(), dtype=theano.config.floatX), borrow=True)

    return shared_traindata, T.cast(shared_trainlabels, 'int32'), shared_testdata, T.cast(shared_testlabels, 'int32')


def run(subBatchSize=500, maxEpochNum=100, eta=0.1, trainErrPeriod=5, testErrPeriod=10,
        logfile='./log.txt', saveWeightFile = None, saveWeightsFor='train', loadWeightFile=None):


    my = myCNN()
    # Read dataset
    #dataset_dir = 'C:\\Users\\Abdulkadir\\Desktop\\Datasets\\mnist\\'
    dataset_dir = './datasets/'
    dataset_filename = 'mnist.pkl.gz'
    (trainImages, trainLabels, testImages, testLabels) = load_mnist_dataset(dataset_dir, dataset_filename)


    # Get the number of images in the training set
    numOfTrainImages = trainImages.get_value().shape[0]
    # Get the number of images in the test set
    numOfTestImages = testImages.get_value().shape[0]
    # Get the sub batch size for training set
    assert (numOfTrainImages % subBatchSize == 0), "The subbatch size must be a divisor of the number of train images"
    numOfTrainSubBatches = numOfTrainImages / subBatchSize
    # Get the sub batch size for test set
    assert (numOfTestImages % subBatchSize == 0), "The subbatch size must be a divisor of the number of test images"
    numOfTestSubBatches = numOfTestImages / subBatchSize


    x = T.matrix('x') # data input symbolic variable
    y = T.ivector('y') # labels symbolic variable

    # -----< Construction of Network Model >-----

    layer0 = x.reshape((subBatchSize, 1, 28, 28))

    [layer1, layer1_w, layer1_b] = my.convolutionLayer(featureMaps=layer0,
                                                       featureMapShape=(subBatchSize, 1, 28, 28),
                                                       kernelShape=(20, 1, 5, 5),
                                                       bias=0.1
                                                       )
    layer2 = my.maxPoolingLayer(featureMaps=layer1,
                                poolingShape=(2, 2),
                                stride=2
                                )


    layer3 = my.reLuLayer(featureMaps=layer2)

    [layer4, layer4_w, layer4_b] = my.convolutionLayer(featureMaps=layer3,
                                                       featureMapShape=(subBatchSize, 20, 12, 12),
                                                       kernelShape=(50, 20, 5, 5)
                                                       )


    layer5 = my.maxPoolingLayer(featureMaps=layer4,
                                poolingShape=(2, 2),
                                stride=2
                                )

    layer6 = my.reLuLayer(featureMaps=layer5)


    layer7 = layer6.flatten(2)

    [layer8, layer8_w, layer8_b] = my.fullyConnectedLayer(inputUnits=layer7,
                                                             inputDim=50*4*4,
                                                             outputDim=500
                                                             )


    layer8 = layer8.reshape((subBatchSize, 500))

    [error, numOfWrongClass, layer9_w, layer9_b] = my.softmaxLayer(inputVect=layer8,
                                                                     labels=y,
                                                                     inputDim=500,
                                                                     numOfClasses=10
                                                                     )


    # --------------------< Construction of Training Function >--------------------
    loadweight = True
    if loadweight is True and loadWeightFile is not None:
        with open(loadWeightFile, 'rb') as w:
            weights = pickle.load(w)
            (param1, param2, param3, param4, param5, param6, param7, param8) = weights
            layer1_w.set_value(param1)
            layer1_b.set_value(param2)
            layer4_w.set_value(param3)
            layer4_b.set_value(param4)
            layer8_w.set_value(param5)
            layer8_b.set_value(param6)
            layer9_w.set_value(param7)
            layer9_b.set_value(param8)
        loadweight = False
        print "Pretrained weights were loaded!"

    # Define symbolic index variable
    index = T.iscalar('index')
    # Define parameters
    params = [layer1_w, layer1_b, layer4_w, layer4_b, layer8_w, layer8_b, layer9_w, layer9_b]
    # Take the derivative of error function with respect to parameters
    grads = T.grad(cost=error, wrt=params)

    # Define updates
    updates = [(w, w - eta * delta) for w, delta in zip(params, grads)]

    # Definition of symbolic training function
    training = function([index],error,
                        givens={
                            x: trainImages[index * subBatchSize: (index + 1) * subBatchSize],
                            y: trainLabels[index * subBatchSize: (index + 1) * subBatchSize]
                        },
                        updates=updates,
                )

    # Definiton of the symbolic function computing the training error
    computeTrainingError = function(
        [index],
        numOfWrongClass,
        givens={
            x: trainImages[index * subBatchSize: (index + 1) * subBatchSize],
            y: trainLabels[index * subBatchSize: (index + 1) * subBatchSize]
        }
    )

    # Definiton of the symbolic testing function
    testing = function(
        [index],
        numOfWrongClass,
        givens={
            x: testImages[index * subBatchSize: (index + 1) * subBatchSize],
            y: testLabels[index * subBatchSize: (index + 1) * subBatchSize]
        }
    )

    print "The total number of training images in the dataset : " + str(numOfTrainImages)
    print "The total number of test images in the dataset : " + str(numOfTestImages)
    # Log file

    with open(logfile, "a") as logf:
        logf.write('The total number of training images in the dataset : ' + str(numOfTrainImages) + '\n')
        logf.write('The total number of test images in the dataset : ' + str(numOfTestImages) + '\n')

    minErr = numOfTrainImages+numOfTestImages

    for epoch in range(1, maxEpochNum+1):

        for subBatchIndex in range(numOfTrainSubBatches):

             err = training(subBatchIndex)

        if (epoch % trainErrPeriod == 0) or (epoch == 1):
            # Compute the training error
            trainingError = [computeTrainingError(inx) for inx in range(numOfTrainSubBatches)]

            # Get the total wrong classified number of elements in the training set
            totalWrongClass = np.sum(trainingError)
            print "Epoch : " + str(epoch) + " Training error : %" + str(totalWrongClass*100.0 / numOfTrainImages) + " " + str(totalWrongClass)
            # Write log file
            with open(logfile, "a") as logf:
                logf.write('Epoch : ' + str(epoch) + '\n')
                logf.write('Training : ' + str(totalWrongClass*100.0 / numOfTrainImages) + ' ' + str(totalWrongClass) + '\n')

        if (epoch % testErrPeriod == 0) or (epoch == 1):
            # Compute the testing error
            testingError = [testing(inx) for inx in range(numOfTestSubBatches)]
            # Get the total wrong classified number of elements in the test set
            totalTestWrongClass = np.sum(testingError)
            print "\t\t  Testing error : %" + str(totalTestWrongClass*100.0 / numOfTestImages) + " " + str(totalTestWrongClass)
            # Write log file
            with open(logfile, "a") as logf:
                logf.write('Testing : ' + str(totalTestWrongClass * 100.0 / numOfTestImages) + ' ' + str(totalTestWrongClass) + '\n')

            if saveWeightsFor == 'train':
                currentErr = totalWrongClass
            elif saveWeightsFor == 'test':
                currentErr = totalTestWrongClass
            else:
                print "Please enter the option name to save weights for training or test!"

            if minErr > currentErr and saveWeightFile is not None:
                print "Weights are saved!"
                minErr = currentErr
                with open(saveWeightFile, 'wb') as w:
                    pickle.dump((layer1_w.get_value(), layer1_b.get_value(),
                                  layer4_w.get_value(), layer4_b.get_value(),
                                  layer8_w.get_value(), layer8_b.get_value(),
                                  layer9_w.get_value(), layer9_b.get_value()),
                                  w, protocol=pickle.HIGHEST_PROTOCOL)


run(subBatchSize=200, trainErrPeriod=1, eta=0.1, testErrPeriod=1, maxEpochNum=500,
    logfile='./mnist_log.txt', saveWeightsFor='test')