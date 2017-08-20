import tensorflow as tf
import numpy as np
import math
n = 19
# images is a 1 x 10 x 10 x 1 array that contains the numbers 1 through 100 in order
images = [[[[x * n + y + 1] for y in range(n)] for x in range(n)]]

# x = np.linspace(1, n, n)
# y = np.linspace(1, n, n)
# z = np.linspace(1, n, n)
# xv, yv = np.meshgrid(x, y);
# II = np.expand_dims(xv, axis=0)#,axis=-0)
# II = np.expand_dims(II, axis=-1)
# images = II
sigma = 2
K = math.ceil(n/sigma)
s = K
with tf.Session() as sess:
    A1 =  tf.extract_image_patches(images=images, ksizes=[1, K, K, 1], strides=[1, s, s, 1], rates=[1, 1, 1, 1], padding='SAME').eval()

print(A1.shape)
print(A1[0,0,0,:])
print(A1[0,0,1,:])
print(A1[0,2,2,:])