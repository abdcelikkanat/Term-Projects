from cnn import myCNN
import theano
import theano.tensor as T
from theano import shared
from theano import function
import numpy as np
from six.moves import cPickle
import sys

import pylab
from PIL import Image


def load_cifar_dataset(dataset_dir, filenames):

    images = np.zeros((0,3072))
    labels = []
    for i in range(len(filenames)):
        print "Loading the dataset"
        with open(dataset_dir+filenames[i], 'rb') as f:
            dataset = cPickle.load(f)

        images_s = dataset['data']
        labels_s = dataset['labels']

        images = np.vstack((images, images_s))
        labels = labels + labels_s

    shared_images = shared(np.asarray(images, dtype=theano.config.floatX), borrow=True)
    shared_labels = shared(np.asarray(labels, dtype=theano.config.floatX), borrow=True)

    return shared_images, T.cast(shared_labels, 'int32')

#img = images[0,:].reshape(3,32,32).transpose(1,2,0)

def run(subBatchSize=500, maxEpochNum=100, eta=0.1, trainErrPeriod=5, testErrPeriod=10, logfile='./log.txt', saveWeightFile = None, loadWeightFile=None):
    loadweight = True

    my = myCNN()
    # Read dataset
    base = './datasets/cifar10'
    trainSet_dir = base + '/'
    train_filename = ('data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5')
    (trainImages, trainLabels) = load_cifar_dataset(trainSet_dir, train_filename)
    testSet_dir = base + '/'
    test_filename = ('test_batch',)
    (testImages, testLabels) = load_cifar_dataset(testSet_dir, test_filename)


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

    layer0 = x.reshape((subBatchSize, 3, 32, 32))

    [layer1, layer1_w, layer1_b] = my.convolutionLayer(featureMaps=layer0,
                                                       featureMapShape=(subBatchSize, 3, 32, 32),
                                                       kernelShape=(32, 3, 5, 5),
                                                       bias=0.1
                                                       )
    layer2 = my.maxPoolingLayer(featureMaps=layer1,
                                poolingShape=(2, 2),
                                stride=2
                                )

    layer3 = my.reLuLayer(featureMaps=layer2)


    [layer4, layer4_w, layer4_b] = my.convolutionLayer(featureMaps=layer3,
                                                       featureMapShape=(subBatchSize, 32, 14, 14),
                                                       kernelShape=(50, 32, 5, 5)
                                                       )

    layer5 = my.maxPoolingLayer(featureMaps=layer4,
                                poolingShape=(2, 2),
                                stride=2
                                )

    layer6 = my.reLuLayer(featureMaps=layer5)


    [layer7, layer7_w, layer7_b] = my.convolutionLayer(featureMaps=layer6,
                                                       featureMapShape=(subBatchSize, 50, 5, 5),
                                                       kernelShape=(64, 50, 5, 5)
                                                       )

    layer8 = my.maxPoolingLayer(featureMaps=layer7,
                                poolingShape=(1, ),
                                stride=2
                                )
    layer8, maxlocsLayer8 = my.maxPoolingLayer(featureMaps=layer7, featureMapShape=(subBatchSize,64, 1, 1), poolingSize=1)

    layer9 = my.reLuLayer(featureMaps=layer8)





    layer10 = layer9.flatten(2)

    layer13 = layer10.reshape((subBatchSize, 1*1*64))

    [error, numOfWrongClass, layer14_w, layer14_b] = my.softmaxLayer(inputVect=layer13,
                                                                     labels=y,
                                                                     inputDim=1*1*64,
                                                                     numOfClasses=10
                                                                     )


    # --------------------< Construction of Training Function >--------------------
    if loadweight is True and loadWeightFile is not None:
        with open(loadWeightFile, 'rb') as w:
            dataset = cPickle.load(w)
            (param1, param2, param3, param4, param5, param6, param7) = dataset
            layer1_w.set_value(param1)
            layer1_b.set_value(param2)
            layer4_w.set_value(param3)
            layer4_b.set_value(param4)
            layer7_w.set_value(param5)
            layer7_b.set_value(param6)
            print "Epoch : " + str(param7)
        loadweight = False
        print "Pretrained weights were loaded!"

    # Define symbolic index variable
    index = T.iscalar('index')
    # Define parameters
    params = [layer1_w, layer1_b, layer4_w, layer4_b, layer14_w, layer14_b]
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

    minTrainErr = numOfTrainImages
    minTestErr = numOfTestImages

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

            if minTrainErr > totalWrongClass and saveWeightFile is not None:
                print "Weights are saved!"
                minTrainErr = totalWrongClass
                with open(saveWeightFile, 'wb') as w:
                    cPickle.dump((layer1_w.get_value(), layer1_b.get_value(),
                                  layer4_w.get_value(), layer4_b.get_value(),
                                  layer7_w.get_value(), layer7_b.get_value(),
                                  epoch), w, protocol=cPickle.HIGHEST_PROTOCOL)


run(subBatchSize=500, trainErrPeriod=1, eta=0.001, testErrPeriod=1, maxEpochNum=500,
    logfile='./log_cifar.txt', saveWeightFile='./weights_cifar.pkl')