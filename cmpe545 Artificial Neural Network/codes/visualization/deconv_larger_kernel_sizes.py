import sys
sys.path.insert(0, '../')
from cnn import myCNN
from decnn import myDECNN
import theano
import theano.tensor as T
from theano import shared
from theano import function
import numpy as np
import matplotlib.pyplot as plt
import six.moves.cPickle as pickle
from PIL import Image
import sys

import pylab
from PIL import Image


def load_pet_dataset(dataset_dir, filenames):
    images = np.zeros((0,12288))
    labels = []
    for i in range(len(filenames)):
        print "Loading the dataset"
        with open(dataset_dir+filenames[i], 'rb') as f:
            subset = pickle.load(f)

        images_t, labels_t = subset

        images = np.vstack((images, images_t))
        labels = labels + labels_t.tolist()

    shared_images = shared(np.asarray(images, dtype=theano.config.floatX), borrow=True)
    shared_labels = shared(np.asarray(labels, dtype=theano.config.floatX), borrow=True)

    return shared_images, T.cast(shared_labels, 'int32')



def run(subBatchSize=5, index=0, loadWeightFile=None):
    loadweight = True

    my = myCNN()
    myd = myDECNN()
    # Read dataset
    base = 'C:/Users/Abdulkadir/Desktop/Datasets/Pet Dataset/10class'
    dir = base + '/'
    filename = ('subset1.pkl',)
    (images, labels) = load_pet_dataset(dir, filename)


    x = T.matrix('x') # data input symbolic variable
    y = T.ivector('y') # labels symbolic variable

    # -----< Construction of Network Model >-----
    layer0 = x.reshape((subBatchSize, 64, 64, 3)).transpose(0, 3, 1, 2)

    [layer1, layer1_w, layer1_b] = my.convolutionLayer(featureMaps=layer0,
                                                       featureMapShape=(subBatchSize, 3, 64, 64),
                                                       kernelShape=(16, 3, 15, 15),
                                                       bias=0.1
                                                       )

    layer2, maxlocsLayer2 = myd.maxPoolingLayer(featureMaps=layer1, featureMapShape=(subBatchSize, 16, 50, 50),
                                                 poolingSize=2)

    layer3 = my.reLuLayer(featureMaps=layer2)

    [layer4, layer4_w, layer4_b] = my.convolutionLayer(featureMaps=layer3,
                                                       featureMapShape=(subBatchSize, 32, 25, 25),
                                                       kernelShape=(32, 16, 11, 11)
                                                       )

    layer5, maxlocsLayer5 = myd.maxPoolingLayer(featureMaps=layer4, featureMapShape=(subBatchSize, 32, 15, 15),
                                                 poolingSize=3)

    layer6 = my.reLuLayer(featureMaps=layer5)

    layer7 = layer6.flatten(2)

    [layer8, layer8_w, layer8_b] = my.fullyConnectedLayer(inputUnits=layer7,
                                                          inputDim=32 * 5 * 5,
                                                          outputDim=64
                                                          )

    layer8 = layer8.reshape((subBatchSize, 64))

    [error, numOfWrongClass, layer9_w, layer9_b] = my.softmaxLayer(inputVect=layer8,
                                                                   labels=y,
                                                                   inputDim=64,
                                                                   numOfClasses=10
                                                                   )


    delayer5 = myd.reLuLayer(featureMaps=layer6)

    delayer4 = myd.maxUnpoolingLayer(input=delayer5, featureMapShape=(subBatchSize, 32, 15, 15),
                                      maxLocations=maxlocsLayer5)

    delayer3, delayer3_w = myd.deconvolutionLayer(featureMaps=delayer4,
                                                   featureMapShape=(subBatchSize, 32, 15, 15),
                                                   kernelShape=(16, 32, 11, 11),
                                                   w=layer4_w)

    delayer2 = myd.reLuLayer(featureMaps=delayer3)

    delayer1 = myd.maxUnpoolingLayer(input=delayer2, featureMapShape=(subBatchSize, 16, 50, 50),
                                      maxLocations=maxlocsLayer2)

    delayer0, delayer0_w = myd.deconvolutionLayer(featureMaps=delayer1,
                                                   featureMapShape=(subBatchSize, 16, 50, 50),
                                                   kernelShape=(3, 32, 15, 15),
                                                   w=layer1_w)


    # --------------------< Construction of Training Function >--------------------
    if loadweight is True and loadWeightFile is not None:
        with open(loadWeightFile, 'rb') as w:
            dataset = pickle.load(w)
            (param1, param2, param3, param4, param5, param6, param7, param8) = dataset
            layer1_w.set_value(param1)
            layer1_b.set_value(param2)
            layer4_w.set_value(param3)
            layer4_b.set_value(param4)
        loadweight = False
        print "Pretrained weights were loaded!"

        # Define symbolic index variable
        index = T.iscalar('index')


    return (layer0.eval({x: images.get_value()[0:subBatchSize]}),
            delayer0.eval({x: images.get_value()[0:subBatchSize]}))


layer0, delayer0 = run(subBatchSize=15, index=0, loadWeightFile='../log/weights_larger_kernels_pet_10class.pkl')


def scaleData(input):
    lin = input.flatten()
    mx = lin.max()
    mn = lin.min()
    output = (input - mn) / (mx - mn)
    return output

# Show the images in the grid form
def gridView(images, image_shape, grid_form, image_type='uint8', numOfChannels=1):
    numOfImages = images.shape[0]
    h, w = image_shape

    if numOfChannels == 1:
        grid_size = (h*grid_form[0], w*grid_form[1])
    else:
        grid_size = (h * grid_form[0], w * grid_form[1], numOfChannels)
    grid = np.zeros(grid_size, dtype=image_type)

    #Scale images
    images = scaleData(images)

    for i in range(grid_form[0]):
        for j in range(grid_form[1]):
            if numOfChannels == 1:
                grid[h*i:h*(i+1), w*j:w*(j+1)] = images[i*grid_form[1]+j].reshape(h,w)
            else:
                grid[h*i:h*(i+1), w*j:w*(j+1),:] = images[i*grid_form[1]+j].reshape(h,w, numOfChannels)

    return grid


grid1 = gridView(layer0.transpose(0,2,3,1), image_shape=(64,64), grid_form=(3,5), image_type=layer0.dtype, numOfChannels=3)
grid2 = gridView(delayer0.transpose(0,2,3,1), image_shape=(64,64), grid_form=(3,5), image_type=layer0.dtype, numOfChannels=3)

grid = np.hstack((grid1, grid2))
plt.imshow(grid)
plt.savefig('./deconv.jpg')
