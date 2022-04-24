import numpy as np


def instance_dimension_reduction(data_set):
    for index in range(data_set[b'data'].shape[0]):
        img = data_set[b'data'][index]
        img_r = img[0:1024].reshape(32, 32)
        img_g = img[1024:2048].reshape(32, 32)
        img_b = img[2048:].reshape(32, 32)
        img_r_new = np.zeros((32, 32))
        img_g_new = np.zeros((32, 32))
        img_b_new = np.zeros((32, 32))
        for i in range(31):
            for j in range(31):
                r = np.average([img_r[i, j], img_r[i, j + 1], img_r[i + 1, j],
                                img_r[i + 1, j + 1]])
                img_r_new[i, j] = r
                g = np.average([img_g[i, j], img_g[i, j + 1], img_g[i + 1, j],
                                img_g[i + 1, j + 1]])
                img_g_new[i, j] = g
                b = np.average([img_b[i, j], img_b[i, j + 1], img_b[i + 1, j],
                                img_b[i + 1, j + 1]])
                img_b_new[i, j] = b
        new_img = np.concatenate((img_r_new.flatten(), img_g_new.flatten(),
                                  img_b_new.flatten()))
        data_set[b'data'][index] = new_img
    return data_set

