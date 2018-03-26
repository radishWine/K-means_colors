import cv2
import numpy as np
import os
from scipy.cluster.vq import *
import shutil
import colorsys

class imageKmeans(object):
    #初始化
    def __init__(self, k):
        self.basedir = os.path.dirname(os.path.abspath(__file__))  # 项目根目录
        self.imagedir = os.path.join(self.basedir, 'flowers')  # 图片根目录
        self.k = k  #簇的个数
        for i in range(self.k):  # 创建k个文件夹用于存放分类后的图片
            clusterDir = os.path.join(self.basedir, 'cluster-{}'.format(i))
            if not os.path.isdir(clusterDir):
                os.mkdir(clusterDir)

    # 返回图片一维数组
    def _loadImages(self):
        images = os.listdir(self.imagedir)
        imagesfiles = [os.path.join(self.imagedir, image) for image in images]
        return imagesfiles

    # 将h,s,v合并
    def _hsvToL(self, h, s, v):
        OH = 0
        if (h <= 20 or h > 315):
            OH = 0
        if (h > 20 and h <= 40):
            OH = 1
        if (h > 40 and h <= 75):
            OH = 2
        if (h > 75 and h <= 155):
            OH = 3
        if (h > 155 and h <= 190):
            OH = 4
        if (h > 190 and h <= 271):
            OH = 5
        if (h > 271 and h <= 295):
            OH = 6
        if (h > 295 and h <= 315):
            OH = 7

        OS = 0
        if (s >= 0 and s <= 0.2):
            OS = 0
        if (s > 0.2 and s < 0.7):
            OS = 1
        if (s > 0.7 and s <= 1.0):
            OS = 2

        OV = 0
        if (v >= 0 and v <= 0.2):
            OV = 0
        if (v > 0.2 and v <= 0.7):
            OV = 1
        if (v > 0.7 and v <= 1.0):
            OV = 2

        L = 9 * OH + 3 * OS + OV
        assert L >= 0 and L <= 71
        return L

    # 提取图片特征
    def _getColor(self,oneImage):
        image = cv2.imread(oneImage)
        image = cv2.resize(image, (600, 600))
        Row = image.shape[0]  # 行600
        Col = image.shape[1]  # 列600
        vector = [0]*12
        for row in range(Row):
            for col in range(Col):
                b, g, r = image[row][col]
                h, s, v = colorsys.rgb_to_hsv(r / 255., g / 255., b / 255.)
                h *= 360
                L = self._hsvToL(h,s,v)
                vector[L//6] += L   #L//6范围 0~11
        Lsum = sum(vector)
        result = [v*1.0/Lsum for v in vector]
        return result

    #调用Kmeans
    def getCluster(self):
        images = self._loadImages() #图片一维数组(绝对地址)
        Avector = []
        for oneImage in images:
            Vector = self._getColor(oneImage)
            print(Vector)
            Avector.append(Vector)
        image_list = np.reshape(Avector,(len(images),-1))
        cu, fangcha = kmeans(image_list, self.k)
        fenlei, juli = vq(image_list, cu)
        print(fenlei)
        for i in range(len(images)):
            for j in range(self.k):
                if ( fenlei[i] == j ):
                    shutil.copyfile(images[i], os.path.join(os.path.join(self.basedir,'cluster-{}'.format(j)), os.path.basename(images[i])))

#主函数
if __name__ == '__main__':
    imagekmeans = imageKmeans(3)
    imagekmeans.getCluster()
