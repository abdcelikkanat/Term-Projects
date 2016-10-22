import sys
sys.path.insert(0, '../')
from cnn import myCNN
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
    layer2 = my.maxPoolingLayer(featureMaps=layer1,
                                poolingShape=(2, 2),
                                stride=2
                                )

    layer3 = my.reLuLayer(featureMaps=layer2)

    [layer4, layer4_w, layer4_b] = my.convolutionLayer(featureMaps=layer3,
                                                       featureMapShape=(subBatchSize, 32, 25, 25),
                                                       kernelShape=(32, 16, 11, 11)
                                                       )

    layer5 = my.maxPoolingLayer(featureMaps=layer4,
                                poolingShape=(3, 3),
                                stride=3
                                )

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


    # --------------------< Construction of Training Function >--------------------
    if loadweight is True and loadWeightFile is not None:
        with open(loadWeightFile, 'rb') as w:
            dataset = pickle.load(w)
            (param1, param2, param3, param4, param5, param6, param7, param8) = dataset
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

    # Definition of symbolic training function
    visualize = function([index],error,
                        givens={
                            x: images[index * subBatchSize: (index + 1) * subBatchSize],
                            y: labels[index * subBatchSize: (index + 1) * subBatchSize]
                        }
                )


    err = visualize(0)

    return (layer0.eval({x: images.get_value()[0:subBatchSize]}),
            layer1.eval({x: images.get_value()[0:subBatchSize]}),
            layer2.eval({x: images.get_value()[0:subBatchSize]}),
            layer3.eval({x: images.get_value()[0:subBatchSize]}))


img, layer1, layer2, layer3 = run(subBatchSize=15, index=0, loadWeightFile='../log/weights_larger_kernels_pet_10class.pkl')

result = layer3
l0 = result[7]

print l0.shape
print l0.dtype
m = img[7].transpose(1,2,0)
plt.imshow(m)
plt.show()

def scaleData(input):
    lin = input.flatten()
    mx = lin.max()
    mn = lin.min()
    output = (input - mn) / (mx - mn)
    return output


def gridView(images, image_shape, grid_size, image_type='uint8'):
    images = (images)
    numOfImages = images.shape[0]
    h, w = image_shape

    grid = np.zeros((h*grid_size[0], w*grid_size[1]), dtype=image_type)

    #images = scaleData(images)

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            grid[h*i:h*(i+1), w*j:w*(j+1)] = images[i*grid_size[1]+j].reshape(h,w)

    plt.imshow(grid, cmap='gray_r', interpolation='none')
    plt.show()

gridView(l0*255, l0.shape[1:], (4,4), image_type=l0.dtype)