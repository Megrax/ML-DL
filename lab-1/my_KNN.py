import numpy as np
import struct
from numpy import *
import operator

# 训练集文件
train_images_idx3_ubyte_file = 'MNIST_data/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = 'MNIST_data/train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = 'MNIST_data/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = 'MNIST_data/t10k-labels.idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    # 因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(
        fmt_header,
        bin_data,
        offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' %
          (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    # 获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    offset += struct.calcsize(fmt_header)
    print(offset)
    # 图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    fmt_image = '>' + str(image_size) + 'B'
    print(fmt_image, offset, struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    # plt.figure()
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
            print(offset)
        images[i] = np.array(
            struct.unpack_from(fmt_image, bin_data, offset)).reshape(
            (num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def classify(testOne, dataSet, labels, k):
    # 取前 6000 作为训练集
    dataSet = dataSet[:6000]
    dataSetSize = dataSet.shape[0]
    # dataSetSize = 1000
    diffMat = tile(testOne, (dataSetSize, 1))-dataSet
    sqDiffMat = diffMat**2
    # 欧式距离
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # 对训练结果中的欧式距离进行排序
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 由距离最小的k个点通过投票判别出结果
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(
        classCount.items(),
        key=operator.itemgetter(1),
        reverse=True)
    return sortedClassCount[0][0]


# 该函数将图像转化成程序中的矩阵变量
def img2vector(images):
    returnVect = zeros((images.shape[0], images.shape[1]*images.shape[2]))
    for i in range(images.shape[0]):
        returnVect[i] = images[i].flatten()
    return returnVect


def knn(k, train_images, train_labels, test_images, test_labels):
    errorCount = 0.0  # 记录错误个数
    # m = test_images.shape[0]
    m = 1000
    for i in range(m):
        classifierResult = classify(
            test_images[i],
            train_images, train_labels, k)  # 调用k近邻法的分类器函数，进行判决
        # 打印每次的分类结果
        # print(
        #     "the classifier %d came back with:%d, the real answer is: %d" %
        #     (i + 1, classifierResult, test_labels[i]))
        if (classifierResult != test_labels[i]):
            errorCount += 1.0
    print("\nTotal number of errors is: %d" % errorCount)
    print("\nTotal error rate is:%f" % (errorCount/float(m)))

    file_write_obj = open("data.txt", 'a')
    file_write_obj.writelines('K is : ')
    file_write_obj.writelines(str(k))
    file_write_obj.write('\n')
    file_write_obj.writelines('Total number of errors is: ')
    file_write_obj.writelines(str(errorCount))
    file_write_obj.write('\n')
    file_write_obj.writelines('Total error rate is: ')
    file_write_obj.writelines(str(errorCount/float(m)))
    file_write_obj.write('\n ')


if __name__ == '__main__':

    train_images = load_train_images()  # 读取训练图片
    train_labels = load_train_labels()  # 读取训练图片对应的标签
    test_images = load_test_images()  # 读取测试图片
    test_labels = load_test_labels()  # 读取测试图片对应的标签
    train_images1 = img2vector(train_images)  # 将训练图片转换为一维数组
    test_images1 = img2vector(test_images)  # 将测试图片转换为一维数组
    for i in range(1, 10):
        knn(i, train_images1, train_labels, test_images1, test_labels)

    print('done')
