import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
img_dir = 'C:/Users/Abdulkadir/Desktop/Datasets/Pet Dataset - Kopya/final/subset1/Abyssinian/11.jpg'



img = Image.open(img_dir)

f = np.asarray(img,dtype='uint8').flatten('A')
print f.shape
s = f.reshape(64,64,3)
plt.imshow(s)
plt.show()