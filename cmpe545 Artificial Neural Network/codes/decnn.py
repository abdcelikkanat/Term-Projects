import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d, max_pool_2d_same_size
from theano.tensor.nnet import conv2d, softmax
from theano import shared



class myDECNN(object):

    def __init__(self, ):
        self.rng = np.random.RandomState(23455)

    def convolutionLayer(self, featureMaps, featureMapShape, kernelShape, bias=0):
        # featureMapShape -> ( subbatch size, num of feature maps, image height, image width)
        # kernelShape -> (num of feature maps in prev, num of feature maps in next, kernel height, kernel width )

        # Define weights of feature map
        w = theano.shared(value=np.asarray(np.random.normal(0, 0.01, kernelShape), dtype=theano.config.floatX),
                          borrow=True)
        # Define bias weights
        b = theano.shared(value=np.asarray(np.zeros(kernelShape[0]), dtype=theano.config.floatX),
                          borrow=True)
        # Convolve kernels with feature maps
        convOut = conv2d(input=featureMaps, filters=w, filter_shape=kernelShape, input_shape=featureMapShape)
        # Add bias term
        result = convOut + b.dimshuffle('x', 0, 'x', 'x')

        return [result, w, b]

    def maxPoolingLayer(self, featureMaps, featureMapShape, poolingSize):
        (n, f, h, w) = featureMapShape
        numOfSubregions_y = int(h / poolingSize)
        numOfSubregions_x = int(w / poolingSize)

        s = featureMaps.flatten()
        minVal = s.min() - 1
        maxVals = max_pool_2d_same_size(input=featureMaps - minVal, patch_size=(poolingSize, poolingSize))
        # maxLocations = (maxVals > 0).nonzero()
        # maxLocations = T.switch(maxVals > 0, np.arange(1,np.prod(list(featureMapShape))+1,dtype=np.int).reshape(featureMapShape), np.zeros(featureMapShape,dtype=np.int)).nonzero_values()-1
        maxLocations = T.switch(maxVals > 0,
                                np.arange(1, np.prod(list(featureMapShape)) + 1, dtype=np.int).reshape(featureMapShape),
                                np.zeros(featureMapShape, dtype=np.int))
        maxLocations2 = pool_2d(input=maxLocations.reshape(featureMapShape), ds=(poolingSize, poolingSize),
                                ignore_border=True)
        maxLocations2 = maxLocations2.nonzero_values() - 1

        pooledLayer = pool_2d(input=featureMaps, ds=(poolingSize, poolingSize), ignore_border=True)

        return pooledLayer, maxLocations2

    def maxUnpoolingLayer(self, input, featureMapShape, maxLocations):
        unpooled = theano.shared(value=np.zeros(shape=featureMapShape, dtype=theano.config.floatX))
        unpooled = T.set_subtensor(unpooled.flatten()[maxLocations], input.flatten()).reshape(featureMapShape)
        return unpooled

    def deconvolutionLayer(self, featureMaps, featureMapShape, kernelShape, w, b=0):
        # featureMapShape -> ( subbatch size, num of feature maps, image height, image width)
        # kernelShape -> (num of feature maps in prev, num of feature maps in next, kernel height, kernel width )

        # Define weights of feature map

        w = w.transpose([1, 0, 2, 3])
        w = w[:, :, ::-1, ::-1]

        # Convolve kernels with feature maps
        convOut = conv2d(input=featureMaps, filters=w, filter_shape=kernelShape, input_shape=featureMapShape,
                         border_mode='full')

        return convOut, w


    def reLuLayer(self, featureMaps):
        output = T.switch(featureMaps > 0, featureMaps, 0)

        return output