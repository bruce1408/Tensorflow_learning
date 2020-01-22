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

    # ## method 1
    startTime = time.time()
    newPath = "/home/bruce/bigVolumn/Datasets/aptos/train_data_"
    pool = Pool(processes=8)
    pool.apply_async(picResize_, args=(path, newPath))
    pool.close()
    pool.join()
    print("程序总共运行时间 %f 分钟" % ((time.time() - startTime)/60))



