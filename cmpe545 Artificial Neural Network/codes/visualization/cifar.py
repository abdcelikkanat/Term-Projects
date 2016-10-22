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
import sys

import pylab
from PIL import Image


def load_cifar_dataset(dataset_dir, filenames):

    images = np.zeros((0,3072))
    labels = []
    for i in range(len(filenames)):
        print "Loading the dataset"
        with open(dataset_dir+filenames[i], 'rb') as f:
            dataset = pickle.load(f)

        images_s = dataset['data']
        labels_s = dataset['labels']

        images = np.vstack((images, images_s))
        labels = labels + labels_s

    shared_images = shared(np.asarray(images, dtype=theano.config.floatX), borrow=True)
    shared_labels = shared(np.asarray(labels, dtype=theano.config.floatX), borrow=True)

    return shared_images, T.cast(shared_labels, 'int32')



def run(subBatchSize=5, index=0, loadWeightFile=None):
    loadweight = True

    my = myCNN()
    # Read dataset
    base = 'C:/Users/Abdulkadir/Desktop/Datasets/cifar-10-python/cifar-10-batches-py'
    dir = base + '/'
    filename = ('data_batch_1',)
    (images, labels) = load_cifar_dataset(dir, filename)

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
                                poolingShape=(1, 1),
                                stride=1
                                )

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
            dataset = pickle.load(w)
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

    # Definition of symbolic training function
    visualize = function([index],error,
                        givens={
                            x: images[index * subBatchSize: (index + 1) * subBatchSize],
                            y: labels[index * subBatchSize: (index + 1) * subBatchSize]
                        }
                )


    err = visualize(0)

    return (layer0.eval({x: images.get_value()[0:subBatchSize]}),
            layer1.eval({x: images.get_value()[0:subBatchSize]}))


img, result = run(subBatchSize=15, index=0, loadWeightFile='../weights/cifar_weights.pkl')

l0 = result[1]

print img.shape
print img.dtype
m = img[1].reshape(3,32,32).transpose(1,2,0)
plt.imshow(m)
plt.show()

def gridView(images, image_shape, grid_size, image_type='uint8'):
    numOfImages = images.shape[0]
    h, w, d = image_shape

    grid = np.zeros((28*grid_size[0], 28*grid_size[1]), dtype=image_type)

    #images = scaleData(images)

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            grid[h*i:h*(i+1), w*j:w*(j+1)] = images[i*grid_size[1]+j].reshape(28,28)

    plt.imshow(grid, cmap='hot_r')
    plt.show()

gridView(l0*255, (28,28,1), (16,2), image_type=l0.dtype)