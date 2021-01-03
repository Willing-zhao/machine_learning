"""
读取DSN下的数据集
"""
from tqdm import tqdm
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img


class Fer2013(object):
    def __init__(self, folder="../DSN/fer2013"):

        self.folder = folder

    def gen_train(self):
    #产生数据训练
        folder = os.path.join(self.folder, 'Training')

        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_train = []
        y_train = []
        for i in tqdm(range(len(expressions))):
            if expressions[i] == 'contempt':
                continue
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = load_img(os.path.join(expression_folder, images[j]), target_size=(48, 48), color_mode="grayscale")
                img = img_to_array(img)  # 灰度化
                x_train.append(img)
                y_train.append(i)
        x_train = np.array(x_train).astype('float32') / 255.
        y_train = np.array(y_train).astype('int')
        return expressions, x_train, y_train

    def gen_train_no(self):

        folder = os.path.join(self.folder, 'Training')
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_train = []
        y_train = []
        import cv2
        for i in tqdm(range(len(expressions))):
            if expressions[i] == 'contempt':
                continue
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = cv2.imread(os.path.join(expression_folder, images[j]), cv2.IMREAD_GRAYSCALE)
                x_train.append(img)
                y_train.append(i)
        x_train = np.array(x_train)
        y_train = np.array(y_train).astype('int')
        return expressions, x_train, y_train

    def gen_valid(self):

        folder = os.path.join(self.folder, 'PublicTest')
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_valid = []
        y_valid = []
        for i in tqdm(range(len(expressions))):
            if expressions[i] == 'contempt':
                continue
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = load_img(os.path.join(expression_folder, images[j]), target_size=(48, 48), color_mode="grayscale")
                img = img_to_array(img)  # 灰度化
                x_valid.append(img)
                y_valid.append(i)
        x_valid = np.array(x_valid).astype('float32') / 255.
        y_valid = np.array(y_valid).astype('int')
        return expressions, x_valid, y_valid

    def gen_valid_no(self):

       # 产生验证数据

        folder = os.path.join(self.folder, 'PublicTest')
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_train = []
        y_train = []
        import cv2
        for i in tqdm(range(len(expressions))):
            if expressions[i] == 'contempt':
                continue
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = cv2.imread(os.path.join(expression_folder, images[j]), cv2.IMREAD_GRAYSCALE)
                x_train.append(img)
                y_train.append(i)
        x_train = np.array(x_train)
        y_train = np.array(y_train).astype('int')
        return expressions, x_train, y_train

    def gen_test(self):
        # 产生验证集数据
        folder = os.path.join(self.folder, 'PrivateTest')
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_test = []
        y_test = []
        for i in tqdm(range(len(expressions))):
            if expressions[i] == 'contempt':
                continue
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = load_img(os.path.join(expression_folder, images[j]), target_size=(48, 48), color_mode="grayscale")
                img = img_to_array(img)  # 灰度化
                x_test.append(img)
                y_test.append(i)
        x_test = np.array(x_test).astype('float32') / 255.
        y_test = np.array(y_test).astype('int')
        return expressions, x_test, y_test

    def gen_test_no(self):

       # 产生验证数据

        folder = os.path.join(self.folder, 'PrivateTest')
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_train = []
        y_train = []
        import cv2
        for i in tqdm(range(len(expressions))):
            if expressions[i] == 'contempt':
                continue
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = cv2.imread(os.path.join(expression_folder, images[j]), cv2.IMREAD_GRAYSCALE)
                x_train.append(img)
                y_train.append(i)
        x_train = np.array(x_train)
        y_train = np.array(y_train).astype('int')
        return expressions, x_train, y_train


class Jaffe(object):
    #划分Jaffe没有测试数据
    def __init__(self):
        self.folder = '../DSN/jaffe'

    def gen_train(self):
       #产生训练数据

        folder = os.path.join(self.folder, 'Training')
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_train = []
        y_train = []
        for i in tqdm(range(len(expressions))):
            if expressions[i] == 'contempt':
                continue
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = load_img(os.path.join(expression_folder, images[j]), target_size=(48, 48), color_mode="grayscale")
                img = img_to_array(img)  # 灰度化
                x_train.append(img)
                y_train.append(i)
        x_train = np.array(x_train).astype('float32') / 255.
        y_train = np.array(y_train).astype('int')
        return expressions, x_train, y_train

    def gen_train_no(self):

        #产生训练数据
        import cv2
        folder = os.path.join(self.folder, 'Training')
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_train = []
        y_train = []
        for i in tqdm(range(len(expressions))):
            if expressions[i] == 'contempt':
                continue
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = cv2.imread(os.path.join(expression_folder, images[j]), cv2.IMREAD_GRAYSCALE)
                x_train.append(img)
                y_train.append(i)
        x_train = np.array(x_train)
        y_train = np.array(y_train).astype('int')
        return expressions, x_train, y_train

    def gen_data(self):
        #生成划分后的数据集，实际使用需要交叉验证

        _, x, y = self.gen_train()
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2019)
        return x_train, x_test, y_train, y_test


class CK(object):
    """
    划分CK+测试数据
    """
    def __init__(self):
        self.folder = '../DSN/ck+'

    def gen_train(self):

        folder = self.folder
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_train = []
        y_train = []
        for i in tqdm(range(len(expressions))):
            if expressions[i] == 'neutral':
                continue
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = load_img(os.path.join(expression_folder, images[j]), target_size=(48, 48), color_mode="grayscale")
                img = img_to_array(img)  # 灰度化
                x_train.append(img)
                y_train.append(i)
        x_train = np.array(x_train).astype('float32') / 255.
        y_train = np.array(y_train).astype('int')
        return expressions, x_train, y_train

    def gen_train_no(self):
        """
        产生训练数据
        :return:
        """
        import cv2
        folder = self.folder
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_train = []
        y_train = []
        for i in tqdm(range(len(expressions))):
            if expressions[i] == 'neutral':
                continue
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = cv2.imread(os.path.join(expression_folder, images[j]), cv2.IMREAD_GRAYSCALE)
                x_train.append(img)
                y_train.append(i)
        x_train = np.array(x_train)
        y_train = np.array(y_train).astype('int')
        return expressions, x_train, y_train

    def gen_data(self):
        _, x, y = self.gen_train()
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2019)
        return x_train, x_test, y_train, y_test
