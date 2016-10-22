import sys
sys.path.insert(0, '../')
from cnn import myCNN
from decnn import myDECNN
import theano
import theano.tensor as T
from theano import shared
from theano import function
import numpy as np

from six.moves import cPickle

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

def model(imageId, weightFile=None):

    my = myCNN()
    myd = myDECNN()
    # Read dataset
    trainSet_dir = 'C:\\Users\\Abdulkadir\\Desktop\\Datasets\\cifar-10-python\\cifar-10-batches-py\\'
    train_filename = ('data_batch_1',)
    (trainImages, trainLabels) = load_cifar_dataset(trainSet_dir, train_filename)
    (images, labels) = (trainImages, trainLabels)

    assert (weightFile is not None), "Please enter a valid weight file!"

    x = T.matrix('x') # data input symbolic variable
    y = T.ivector('y') # labels symbolic variable

    # -----< Construction of Network Model >-----
    subBatchSize = 1
    layer0 = x.reshape((subBatchSize, 3, 32, 32))

    [layer1, layer1_w, layer1_b] = my.convolutionLayer(featureMaps=layer0,
                                                       featureMapShape=(subBatchSize, 3, 32, 32),
                                                       kernelShape=(32, 3, 5, 5),
                                                       bias=0.1
                                                       )

    layer2, maxlocsLayer2 = myd.maxPoolingLayer(featureMaps=layer1, featureMapShape=(subBatchSize, 32, 28, 28), poolingSize=2)

    layer3 = my.reLuLayer(featureMaps=layer2)


    [layer4, layer4_w, layer4_b] = my.convolutionLayer(featureMaps=layer3,
                                                       featureMapShape=(subBatchSize, 32, 14, 14),
                                                       kernelShape=(50, 32, 5, 5)
                                                       )
    layer5, maxlocsLayer4 = myd.maxPoolingLayer(featureMaps=layer4, featureMapShape=(subBatchSize, 50, 10, 10), poolingSize=2)

    layer6 = my.reLuLayer(featureMaps=layer5)


    [layer7, layer7_w, layer7_b] = my.convolutionLayer(featureMaps=layer6,
                                                       featureMapShape=(subBatchSize, 50, 5, 5),
                                                       kernelShape=(64, 50, 5, 5))

    layer8, maxlocsLayer8 = myd.maxPoolingLayer(featureMaps=layer7,
                                                featureMapShape=(subBatchSize, 64, 1, 1),
                                                poolingSize=1)

    layer9 = my.reLuLayer(featureMaps=layer8)


    layer10 = layer9.flatten(2)

    layer13 = layer10.reshape((subBatchSize, 1 * 1 * 64))

    [error, numOfWrongClass, layer14_w, layer14_b] = my.softmaxLayer(inputVect=layer13,
                                                                     labels=y,
                                                                     inputDim=1 * 1 * 64,
                                                                     numOfClasses=10
                                                                   )

    T.set_subtensor(layer9[0,:1,:,:], 0)
    T.set_subtensor(layer9[0,1:,:,:], 0)


    delayer8 = myd.reLuLayer(featureMaps=layer9)

    delayer7 = myd.maxUnpoolingLayer(input=delayer8, featureMapShape=(subBatchSize, 64, 1, 1),
                                      maxLocations=maxlocsLayer8)

    delayer6, delayer6_w = myd.deconvolutionLayer(featureMaps=delayer7,
                                                   featureMapShape=(subBatchSize, 64, 1, 1),
                                                   kernelShape=(50, 64, 5, 5),
                                                   w=layer7_w)

    delayer5 = myd.reLuLayer(featureMaps=delayer6)

    delayer4 = myd.maxUnpoolingLayer(input=delayer5, featureMapShape=(subBatchSize, 50, 10, 10),
                                      maxLocations=maxlocsLayer4)


    delayer3, delayer3_w = myd.deconvolutionLayer(featureMaps=delayer4,
                                                   featureMapShape=(subBatchSize, 50, 10, 10),
                                                   kernelShape=(32, 50, 5, 5),
                                                   w=layer4_w)

    delayer2 = myd.reLuLayer(featureMaps=delayer3)

    delayer1 = myd.maxUnpoolingLayer(input=delayer2, featureMapShape=(subBatchSize, 32, 28, 28),
                                      maxLocations=maxlocsLayer2)


    delayer0, delayer0_w = myd.deconvolutionLayer(featureMaps=delayer1,
                                                   featureMapShape=(subBatchSize, 32, 28, 28),
                                                   kernelShape=(3, 32, 5, 5),
                                                   w=layer1_w)



    # --------------------< Construction of Training Function >--------------------

    with open(weightFile, 'rb') as w:
        dataset = cPickle.load(w)
        (param1, param2, param3, param4, param5, param6, param7) = dataset
        layer1_w.set_value(param1)
        layer1_b.set_value(param2)
        layer4_w.set_value(param3)
        layer4_b.set_value(param4)
        layer7_w.set_value(param5)
        layer7_b.set_value(param6)
        print "Epoch : " + str(param7)
        print "Pretrained weights were loaded!"

    # Define symbolic index variable
    index = T.iscalar('index')

    # Definiton of the symbolic function computing the prediction
    prediction = function(
        [index],
        numOfWrongClass,
        givens={
            x: images[index:index+1],
            y: labels[index:index+1]
        }
    )

     # Compute the testing error
    predictionError = prediction(imageId)
    print "The class of the predicted image is true or false :" + str(predictionError)

    return (layer0.eval({x: images.get_value()[imageId:imageId+1]}),
            layer1.eval({x: images.get_value()[imageId:imageId + 1]}),
            layer8.eval({x: images.get_value()[imageId:imageId + 1]}),
            delayer0.eval({x: images.get_value()[imageId:imageId + 1]}))


import matplotlib.pyplot as plt
"""
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
"""

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)
    plt.show()
    plt.axis('off')

def scaleIt(a):
    f = a.flatten()
    mx = f.max()
    mn = f.min()
    f = (f - mn) / (mx - mn)
    return f.reshape(3,32,32)


params = model(7, weightFile='../../deconv/cifarnet_weights.pkl')
(layer0, layer1, layer8, delayer8) = params

img = delayer8[0]
img = scaleIt(img.reshape(3,32,32)).transpose(1,2,0)
plt.imshow((img))
plt.show()

#print params.shape
#image = params.reshape((3, 64, 64)).transpose(1, 2, 0)*256.0

image = layer0.reshape((3,32,32)).transpose(1,2,0)
plt.imshow(image)
#plt.show()

#vis_square(layer1[0,0:16,:,:])



r = delayer8.reshape(3,32,32)
f = np.zeros((3,32,32))
#f[0,:,:] = scaleIt(r[0,:,:])
#f[1,:,:] = scaleIt(r[1,:,:])
#f[2,:,:] = scaleIt(r[2,:,:])
f = scaleIt(r)


plt.imshow(f.transpose(1,2,0))
plt.show()











def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X[0].dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(out_shape,
                        dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] = this_img * c
        return out_array

print delayer8.shape
print "--------"
print delayer8.dtype
print layer0.dtype
print"----------"
input = np.zeros((25, 3, 32, 32))
output = np.zeros((25, 3, 32, 32))
for i in range(25):
    input[i] = layer0.reshape((1, 3,32,32))
    output[i] = delayer8.reshape((1, 3,32,32))


# from bc01 to cb01
input = np.transpose(input, [1, 0, 2, 3])
output = np.transpose(output, [1, 0, 2, 3])

# flatten
input = input.reshape([3, 25, 32 * 32])
output = output.reshape([3, 25, 32 * 32])

# transform to fit tile_raster_images
input = tuple([input[i] for i in xrange(3)] + [None])
output = tuple([output[i] for i in xrange(3)] + [None])

input_map = tile_raster_images(input, img_shape=(32, 32), tile_shape=(5, 5),
                               tile_spacing=(1, 1), scale_rows_to_unit_interval=True,
                               output_pixel_vals=False)

output_map = tile_raster_images(output, img_shape=(32, 32), tile_shape=(5, 5),
                                tile_spacing=(1, 1), scale_rows_to_unit_interval=True,
                                output_pixel_vals=False)

bigmap = np.append(input_map, output_map, axis=1)

plt.imshow(bigmap)
#plt.show()

