from tqdm import tqdm
from PIL import Image
from tqdm import tqdm
from multiprocessing import Process
import natsort as nst
from multiprocessing import Lock, Manager, Process, Queue
from multiprocessing import Pool
import os, time, random
MODE = 'folder'  # or 'file', if you choose a plain text file (see above).
DATASET_PATH = '/home/bruce/bigVolumn/Datasets/aptos/train_images'  # the dataset file or root folder path.
pathNew = "/home/bruce/bigVolumn/Datasets/aptos/train_data"
path = "/home/bruce/bigVolumn/Datasets/aptos/train_images"

# Image Parameters
N_CLASSES = 5  # CHANGE HERE, total number of classes
IMG_WIDTH = 1736  # CHANGE HERE, the image width to be resized to
IMG_HEIGHT = 1736  # CHANGE HERE, the image height to be resized to
CHANNELS = 3  # The 3 color channels, change to 1 if grayscale

def picResize(filename):
    fileWPath = os.path.join(path, filename)
    img = Image.open(fileWPath)
    img = img.resize((1736, 1736), Image.ANTIALIAS)
    img.save(os.path.join(pathNew, filename))
    print(os.path.join(pathNew, filename))


def picResize_(scrPath, tarPath):
    filename = os.listdir(scrPath)
    for filepath in filename:
        fileWPath = os.path.join(path, filepath)
        img = Image.open(fileWPath)
        img = img.resize((1736, 1736), Image.ANTIALIAS)
        img.save(os.path.join(tarPath, filepath))
        print(os.path.join(tarPath, filepath))


if __name__ == "__main__":
    fileList = list()
    filePath = os.listdir(path)
    picNum = len(filePath)
    for i in tqdm(filePath):
        fileList.append(i)
    print(fileList.__len__())
    # ## method 1
    startTime = time.time()
    m = Manager().dict()  # 从这个类中获取Queue类
    m['count'] = 0
    pool = Pool(processes=8)
    for i in tqdm(range(picNum)):
        pool.apply_async(picResize, (fileList[i], ))
    pool.close()
    pool.join()
    print("程序总共运行时间 %f 小时" % ((time.time() - startTime)/60))

    newPath = "/home/bruce/bigVolumn/Datasets/aptos/train_data_"
    pool = Pool(processes=8)
    pool.apply_async(picResize_, args=(path, newPath))



# import pandas as pd
# df = pd.read_csv("/home/bruce/bigVolumn/Datasets/aptos/train.csv")
# df.id_code = pathNew + df.id_code.apply(str) + '.png'
# imagepaths = df.id_code.tolist()
# labels = df['diagnosis'].tolist()
# print(imagepaths)
# print(labels)
#
# assert len(imagepaths) == len(labels)
# num_examples = len(imagepaths)
# n_truct = 3600
# num_train = int(0.8 * 3600)  # 2880
#
#
# print('the num train is:', num_train)
# trainData = imagepaths[:num_train]  # 0-2880
# print(trainData.__len__())
# trainLabel = labels[:num_train]  # 0-2880
# print(trainLabel.__len__())
# testData = imagepaths[num_train:3600]  #
# print(testData.__len__())
# testLabel = labels[num_train:3600]
# print(testLabel.__len__())