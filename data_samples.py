# coding=utf-8
import os
import tensorflow as tf
import numpy
import json
from PIL import Image, ImageDraw


PUBG_ROOT = 'train_source/pubg_train'
PUBG_IMG_DIRS = ['pubgm1/1/', 'pubgm1/2/', 'pubgm1/3/', 'pubgm1/4/', 'pubgm2/1/', 'pubgm2/2/', 'pubgm2/3/']
PUBG_IMG_LABELS = ['pubgm1_1.json', 'pubgm1_2.json', 'pubgm1_3.json', 'pubgm1_4.json', 'pubgm2_1.json', 'pubgm2_2.json', 'pubgm2_3.json']
CLASSIFY_PATH = PUBG_ROOT + '/classes.txt'
CLASSIFY_TABLE = {}
OUT_PUT_DIR = PUBG_ROOT + '/out'


def init_classify_table():
    fo = open(CLASSIFY_PATH, 'r')
    index = -1
    for line in fo.readlines():
        index += 1
        line = line.strip()
        if len(line) > 0:
            CLASSIFY_TABLE[line] = index
    return


def join_features(boxes, labels):
    boxes = tf.stack(boxes, axis=0)
    labels = tf.cast(tf.stack([labels], axis=1), tf.float32)
    return tf.concat([boxes, labels], axis=1)


def get_image_features(img_path, feature_path):
    features = json.loads(open(feature_path).read())

    # shape
    with Image.open(img_path) as img:
        shape = [img.width, img.height, 3]

    # 获取每个object的信息
    bboxes = []
    labels_index = []
    labels_text = []
    for obj in features['outputs']['object']:
        name = obj['name']
        index = CLASSIFY_TABLE[name]
        labels_text.append(name)
        labels_index.append(index)

        bbox = obj['bndbox']
        bboxes.append((float(bbox['xmin']) / shape[0],
                       float(bbox['ymin']) / shape[1],
                       float(bbox['xmax']) / shape[0],
                       float(bbox['ymax']) / shape[1]
                       ))

    joined = join_features(bboxes, labels_index)
    return joined, shape, bboxes, labels_index, labels_text


def pubg_train_data():
    print('\n>> Begin converting')
    if len(PUBG_IMG_DIRS) > len(PUBG_IMG_LABELS):
        print("PUBG_IMG_DIRS and PUBG_IMG_LABELS have different size")
        return

    init_classify_table()
    feature_index = -1

    x_trains = []
    y_trains = []
    for cur_dir in PUBG_IMG_DIRS:
        feature_index += 1
        feature_path = os.path.join(PUBG_ROOT, PUBG_IMG_LABELS[feature_index])
        sub_dir = os.path.join(PUBG_ROOT, cur_dir)
        sub_dir_files = tf.io.gfile.listdir(sub_dir)
        for img_file in sub_dir_files:
            img_path = os.path.join(sub_dir, img_file)
            joined, _, _, _, _ = get_image_features(img_path, feature_path)
            x_trains.append(img_path)
            y_trains.append(joined)
            # testing
            # data_test(img_path, joined)
    print('\n>> Finished converting')
    return x_trains, y_trains


def pubg_data_normalization(index, x_trains, y_trains):
    y_item = y_trains[index]
    return x_trains[index], y_item
    return index, index


def pubg_train_data_set():
    x_trains, y_trains = pubg_train_data()
    y_tensor = tf.convert_to_tensor(y_trains)
    # training_data_set = tf.data.Dataset.range(len(x_trains)).\
    #     map(lambda x: pubg_data_normalization(x.eval(session=tf.compat.v1.Session), x_trains, y_trains))

    # training_data_set = tf.data.Dataset.from_tensor_slices([x_trains, y_trains])

    return training_data_set


# ############################### just for testing###########################################
IMG_COUNT = 0


def data_test(img_path, joined_features):
    global IMG_COUNT
    IMG_COUNT += 1
    if not tf.io.gfile.exists('./test/'):
        tf.io.gfile.mkdir('./test')
    img = Image.open(img_path, 'r')
    img_width = img.width
    img_height = img.height
    draw = ImageDraw.Draw(img)
    size = joined_features.get_shape()
    for i in range(size[0]):
        draw.rectangle((joined_features[i][0]*img_width,
                        joined_features[i][1]*img_height,
                        joined_features[i][2]*img_width,
                        joined_features[i][3]*img_height),
                       outline='red', width=3)
        draw.text((joined_features[i][0]*img_width, joined_features[i][1]*img_height),
                  "{}".format(joined_features[i][4]))
    img.save('./test/{}.jpg'.format(IMG_COUNT))
    return
##################################################################


if __name__ == '__main__':
    # init_classify_table()
    pubg_train_data_set()


