import pandas as pd
import numpy as np
import os
import shutil
import random



def do_label(df):

    n = len(df)
    list1 = []
    for i in range(n):
        a = random.random()
        list1.append(a)
    print(list1)
    df['random'] = pd.DataFrame({'label': list1})
    print(df.describe())
    df['label'] = df['random'].apply(lambda x: 1 if x < 0.18 else 0)
    print(df)
    df.to_csv('labels.csv')
    return df

def split_train_test(img_path, df_path, save_path):

    df = pd.read_csv(df_path)

    path1 = os.listdir(img_path)

    path1_path = [img_path + i for i in path1]
    print(path1_path)

    # mkdir train and test file
    train_path = os.path.join(save_path, 'train/')
    test_path = os.path.join(save_path, 'test/')
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    print(train_path, test_path)
    for i in range(2):
        train_img = df[df['label'] == i]
        train_id = train_img['num'].tolist()
        if i == 0:
            train_img.to_csv('./data/train.csv')
        else:
            train_img.to_csv('./data/test.csv')

        # copy img to train or test
        for idx in train_id:
            img = img_path + idx + '_ref.jpg'
            if os.path.exists(img):
                if i == 0:
                    shutil.copy(img, train_path)
                    print('success copy to train file')
                if i == 1:
                    shutil.copy(img, test_path)
                    print('success copy to test file')

            else:
                print('{} file not found'.format(img))

    print('finish split dataset')
    # split_train_test_val('D:/Data/node/data')


if __name__ == '__main__':

    img_path = './data/img/'
    df_path = './data/labels.csv'
    save_path = './data/'
    split_train_test(img_path, df_path, save_path)
