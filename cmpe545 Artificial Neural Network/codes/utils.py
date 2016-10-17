import numpy as np
from PIL import Image
import pylab as pl
import os
import six.moves.cPickle as pickle


def convertPickleFile_yedek(root_dir, save_dir, filename, image_size):
    images = np.empty((0, np.prod(image_size)), float)
    labels = np.empty((0, 1), int)
    classNum = 0
    for root, dirs, files in os.walk(root_dir):

        if root != root_dir:
            print "Currently processing the class " + str(classNum)

            for file in files:
                img = Image.open(root_dir + "/" + root.split('/')[-1] + "/" + file)
                imgarr = np.asarray(img, dtype='float32') / 255.0
                img.close()

                images = np.vstack((images, imgarr.flatten()))
                labels = np.vstack((labels, classNum))
            classNum += 1

    print "Now saving the file"
    with open(save_dir + filename, 'wb') as f:
        pickle.dump((images, labels), f, protocol=pickle.HIGHEST_PROTOCOL)

    print "Finished!"

#dataset_dir = 'C:/Users/Abdulkadir/Desktop/Datasets/Pet Dataset/augmented/subset1/'
#save_dir = 'C:/Users/Abdulkadir/Desktop/Datasets/Pet Dataset/'
#filename = 'subset1.pkl'
#convertPickleFile(dataset_dir, save_dir, filename, (128,128,3))


def convertPickleFile(root_dir, save_dir, filename, image_size, numOfClasses):
    dog_classes = ['american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle','boxer',
            'chihuahua', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired',
            'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger',
            'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard',
            'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier',
            'wheaten_terrier', 'yorkshire_terrier']
    cat_classes = ['Abyssinian', 'Bengal', 'Birman', 'Bombay','British_Shorthair', 'Egyptian_Mau',
            'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx']

    all_classes = dog_classes + cat_classes

    images = np.empty((0, np.prod(image_size)), dtype=np.float32)
    #labels = np.empty((0, 1), dtype=np.int)
    labels = []
    classLabel = 0
    for root, dirs, files in os.walk(root_dir):

        if root != root_dir:

            className = root.split('/')[-1]


            # If only the first 'numOfClasses' classes is used
            if numOfClasses > 2 and className in all_classes[0:numOfClasses]:
                print "Currently processing the class " + className

                for file in files:
                    img = Image.open(root_dir + className + "/" + file)
                    imgarr = np.asarray(img, dtype=np.float32) / 255.0
                    img.close()

                    images = np.vstack((images, imgarr.flatten()))
                    labels.append(classLabel)
                classLabel += 1

            # If the classes are only dogs and cats
            elif numOfClasses == 2:
                print "Currently processing the class " + className

                # Set the class labels
                if className in dog_classes:
                    classLabel = 0
                elif className in cat_classes:
                    classLabel = 1
                else:
                    print "Unknown class name!"

                for file in files:
                    img = Image.open(root_dir + className + "/" + file)
                    imgarr = np.asarray(img, dtype=np.float32) / 255.0
                    img.close()

                    images = np.vstack((images, imgarr.flatten()))
                    labels.append(classLabel)

    # Now shuffle the images
    labels = np.asarray(labels)
    N = labels.shape[0] # N is the number of images
    # Make the number of images divisible by 100
    N = N - N%100
    print "Total number of images " + str(N)
    inx = np.arange(N, dtype=np.int)
    np.random.shuffle(inx)
    labels = labels[inx]
    images = images[inx]

    print "Now saving the file"
    with open(save_dir + filename, 'wb') as f:
        pickle.dump((images, labels), f, protocol=pickle.HIGHEST_PROTOCOL)

    print "Finished!"


dataset_dir = 'C:/Users/Abdulkadir/Desktop/Datasets/Pet Dataset/final/subset'
save_dir = 'C:/Users/Abdulkadir/Desktop/Datasets/Pet Dataset/37class/'
filename = 'subset'
for i in np.arange(1,11):
    print "Current subset : " + str(i)
    convertPickleFile(dataset_dir+str(i)+'/', save_dir, filename+str(i)+'.pkl', (64,64,3), 37)


"""

with open('C:/Users/Abdulkadir/Desktop/Datasets/Pet Dataset/2class/subset1.pkl', 'rb') as f:
    subset = pickle.load(f)

images, labels = subset

print labels[90:160]

image = images[11]
pl.imshow(image.reshape(64,64,3))
pl.show()

"""