import pickle
import numpy as np
import cv2

fo = open('data/cifar-10-batches-py/test_batch', 'rb')
data_dict = pickle.load(fo, encoding='bytes')

for j in range(10):
    img = np.reshape(data_dict[b'data'][j], (3, 32, 32))
    img = np.transpose(img, (1, 2, 0))
    # 通道顺序为RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_name = 'cifar10_test' + str(data_dict[b'labels'][j]) + str(j) + '.jpg'
    cv2.imwrite(img_name, img)
