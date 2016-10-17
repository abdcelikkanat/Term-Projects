import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d, max_pool_2d_same_size
from theano.tensor.nnet import conv2d, softmax
from theano import shared



class myCNN(object):

    def __init__(self, ):
        rs = np.random.RandomState(1234)
        self.rng = theano.tensor.shared_randomstreams.RandomStreams(rs.randint(999999))

    def convolutionLayer(self, featureMaps, featureMapShape, kernelShape, bias=0.):
        # featureMapShape -> ( subbatch size, num of feature maps, image height, image width)
        # kernelShape -> (num of feature maps in prev, num of feature maps in next, kernel height, kernel width )

        # Define weights of feature map
        w = theano.shared(value=np.asarray(np.random.normal(0, 0.01, kernelShape), dtype=theano.config.floatX),
                          borrow=True)
        # Define bias weights
        b = theano.shared(value=np.asarray(np.ones(kernelShape[0])*bias, dtype=theano.config.floatX),
                          borrow=True)
        # Convolve kernels with feature maps
        convOut = conv2d(input=featureMaps, filters=w, filter_shape=kernelShape, input_shape=featureMapShape)
        # Add bias term
        result = convOut + b.dimshuffle('x', 0, 'x', 'x')

        return [result, w, b]

    def maxPoolingLayer(self, featureMaps, poolingShape=(2, 2), stride=-1):

        if stride == -1:
            strideShape = poolingShape
        else:
            strideShape = (stride, stride)

        # Max pooling operation
        result = pool_2d(input=featureMaps, ds=poolingShape, st=strideShape, ignore_border=True)

        return result

    def reLuLayer(self, featureMaps):

        output = T.switch(featureMaps > 0, featureMaps, 0)

        return output

    def fullyConnectedLayer(self, inputUnits, inputDim, outputDim, bias=0.):

        # Define weights for units
        w = theano.shared(value=np.asarray(np.random.normal(0, 0.01, (inputDim, outputDim)), dtype=theano.config.floatX),
                          borrow=True)
        # Define bias terms
        b = theano.shared(value=np.ones(outputDim, dtype=theano.config.floatX)*bias,
                          borrow=True)

        outputUnits = T.dot(inputUnits, w) + b

        return [outputUnits, w, b]


    def dropoutLayer(self, inputUnits, inputDim, outputDim, bias=0., prob=0.5):

        # Define weights for units
        w = theano.shared(
            value=np.asarray(np.random.normal(0, 0.01, (inputDim, outputDim)), dtype=theano.config.floatX),
            borrow=True)
        # Define bias terms
        b = theano.shared(value=np.zeros(outputDim, dtype=theano.config.floatX),
                          borrow=True)

        outputUnits = T.dot(inputUnits, w) + b

        mask = self.rng.binomial(size=(outputDim,), p=prob)
        outputUnits = outputUnits * mask

        return [outputUnits, w, b]


    def softmaxLayer(self, inputVect, labels, inputDim, numOfClasses):
        # x -> input images
        # r -> labels

        # Get the number of instances
        N = labels.shape[0]

        w = shared(value=np.zeros((inputDim, numOfClasses), dtype=theano.config.floatX),
                   borrow=True)
        w0 = shared(value=np.zeros(numOfClasses, dtype=theano.config.floatX),
                    borrow=True)

        #
        y = softmax(T.dot(inputVect, w) + w0)

        # Calculate the error
        error = -T.mean(T.log(y[T.arange(N), labels]), axis=0)

        # Calculate the total misclassifications
        predictions = T.argmax(y, axis=1)
        totalMisClass = T.sum(T.neq(predictions, labels))

        return [error, totalMisClass, w, w0]