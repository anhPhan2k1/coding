import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy import signal
import os
import operator
import shutil

class ImageQualityAssessment():
    def __init__(self, image):
        self.image = image
        self.w,self.h,self.d = self.image.shape

    def brightness(self):
        image = self.image / np.max(self.image)
        image *= 255
        image = image.astype(np.uint8)
        imageHSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        imageH, imageS, imageV = cv2.split(imageHSV)
        brightness = np.sum(imageV) / (256 * (self.w * self.h + 1 / 255))
        return brightness

    def contrast(self):
        ill = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        aver = np.average(ill)
        averMatrix = aver * np.ones((self.w, self.h), np.float)
        sub = np.abs(ill - averMatrix)
        subRes = sub.reshape((self.w * self.h,))
        cont = np.dot(subRes, subRes.T)
        cont /= self.w * self.h
        return np.sqrt(cont)

    def sharpness(self):
        gray = 0.299 * self.image[:, :, 0] + 0.587 * self.image[:, :, 1] + 0.114 * self.image[:, :, 2]
        gray = gaussian_filter(gray, sigma=.5)
        kernel = np.zeros((3, 3), np.float)
        kernel[1][1] = -4
        kernel[0][1] = 1
        kernel[1][0] = 1
        kernel[1][2] = 1
        kernel[2][1] = 1
        convoluted = np.absolute(signal.convolve2d(gray, kernel))
        threshold = 10
        crop = 5
        sharpness_result = np.zeros((convoluted.shape[0], convoluted.shape[1]), np.float)
        sharpness_result[np.where(convoluted > threshold)] = 255
        if convoluted.shape[0] > crop * 2 and convoluted.shape[1] > crop * 2:
            ratio = np.sum(sharpness_result[crop:convoluted.shape[0] - crop, crop:convoluted.shape[1] - crop]) /\
                    ((convoluted.shape[0] - crop * 2) * (convoluted.shape[1] - crop * 2) + 1)
        else:
            ratio = np.sum(sharpness_result) / (convoluted.shape[0] * convoluted.shape[1] + 1)
        return 100 * ratio / 255
    
if __name__=="__main__":
    image_dir = "/home/anhp/Documents/coding/analysis/10/image/night/car"
    image_target = "/home/anhp/Documents/coding/analysis/10/image/test"
    image_names = os.listdir(image_dir)
    i = 0 
    for image_name in image_names:
        
        if int(image_name[0]) != 0: 
            shutil.copy(os.path.join(image_dir, image_name), os.path.join(image_target, image_name))
            i += 1
        if i == 100:
            break