import numpy as np
import cv2
import os
from PIL import Image


class DataProcess:

    def __init__(self, img_path, out_path):
        self.img_path = img_path
        self.out_path = out_path

    def raw_to_jpg(self,):

        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)

        for file in os.listdir(self.img_path):
            if file.split('.')[-1] == 'raw':
                raw_img = np.fromfile(os.path.join(self.img_path, file), 'float32')
                format_img = np.zeros((703, 800, 128))
                for row in range(703):
                    for dim in range(128):
                        format_img[row, :, dim] = raw_img[(dim + row * 128) * 800:(dim + 1 + row * 128) * 800]
                imgR = format_img[:, :, 49] * 255
                imgG = format_img[:, :, 30] * 255
                imgB = format_img[:, :, 12] * 255
                rgb_img = cv2.merge([imgR, imgG, imgB])

                path3 = self.out_path + file.split('.')[0] + '.jpg'
                print(path3)
                cv2.imwrite(path3, rgb_img)


def save_resize_img(img_path, img_path_train, w, h):

    img_path_list = os.listdir(img_path)
    for filename in img_path_list:

        file = img_path + filename
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        new_img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        if not os.path.exists(img_path_train):
            os.mkdir(img_path_train)
        save_path = img_path_train + filename
        print(save_path)
        cv2.imwrite(save_path, new_img)



if __name__ == '__main__':

    p1 = './data/raw_data/'
    p2 = './data/img/'
    dataprocess = DataProcess(p1, p2)
    dataprocess.raw_to_jpg()
