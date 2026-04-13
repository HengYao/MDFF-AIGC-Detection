import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
import copy
import os
import imageio
import pickle

def data_augment(img, opt):
    img = np.array(img)
    # print(opt.flag)
    # if int(opt.flag)>0:
    #     if opt.blur_prob<0.2:
    #         opt.blur_prob = opt.blur_prob+0.01
    #         # opt.flag=str(int(opt.flag)-1)
    #         # print('现在后处理的概率为',opt.blur_prob)
    #     if opt.jpg_prob<0.2:
    #         opt.jpg_prob = opt.jpg_prob+0.01

    if random() < opt.jpg_prob:
        a = random()
        if a < 0.33:
          sig = sample_continuous(opt.blur_sig)
          gaussian_blur(img, sig)
        elif a<0.66:
          sig = sample_continuous(opt.blur_sig)
          gaussian_blur_directional(img,sig)
        else :
            sigma_spatial = 50
            sigma_color = 50
            img = gaussian_blur_bilateral(img, sigma_spatial, sigma_color)
    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)
    #
    if random() < opt.jpg_prob:
        random_angle = np.random.uniform(-90, 90)
        img = rotate_image(img, random_angle)
    if random() < opt.jpg_prob:
        img = quantize_image(img, num_bits=5)
    # print('现在后处理的概率为', opt.blur_prob)
    # if random() < opt.blur_prob:
    #     sig = sample_continuous(opt.blur_sig)
    #     gaussian_blur(img, sig)
    #
    # if random() < opt.jpg_prob:
    #     method = sample_discrete(opt.jpg_method)
    #     qual = sample_discrete(opt.jpg_qual)
    #     img = jpeg_from_key(img, qual, method)
    return Image.fromarray(img)


def sample_continuous(s):  #s正常是0和3，也就是在0-3之前随机选择sigma

    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

def gaussian_blur_directional(img, sigma):
    gaussian_filter1d(img[:,:,0], output=img[:,:,0], sigma=sigma, axis=0)
    gaussian_filter1d(img[:,:,0], output=img[:,:,0], sigma=sigma, axis=1)
    gaussian_filter1d(img[:,:,1], output=img[:,:,1], sigma=sigma, axis=0)
    gaussian_filter1d(img[:,:,1], output=img[:,:,1], sigma=sigma, axis=1)
    gaussian_filter1d(img[:,:,2], output=img[:,:,2], sigma=sigma, axis=0)
    gaussian_filter1d(img[:,:,2], output=img[:,:,2], sigma=sigma, axis=1)

def gaussian_blur_bilateral(img, sigma_spatial, sigma_color):
    img_blur = cv2.bilateralFilter(img, -1, sigma_spatial, sigma_color)
    return img_blur


def rotate_image(image, angle):
    # 获取图像尺寸
    height, width = image.shape[:2]
    # 计算旋转中心
    center = (width // 2, height // 2)
    # 执行旋转
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
    return rotated_image


def quantize_image(image, num_bits):
    max_value = 2 ** num_bits - 1
    quantized_image = np.uint8(np.round(image / 255 * max_value) / max_value * 255)
    return quantized_image
def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}

def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)

def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg"]):
    out = []
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out

def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [ item for item in image_list if must_contain in item   ]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list

class read_data():
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        real_img_list = [os.path.join(self.root, '0_real', train_file) for train_file in
                        os.listdir(os.path.join(self.root, '0_real'))]

        real_label_list = [0 for _ in range(len(real_img_list))]

        fake_img_list = [os.path.join(self.root, '1_fake', train_file) for train_file in
                        os.listdir(os.path.join(self.root, '1_fake'))]

        fake_label_list = [1 for _ in range(len(fake_img_list))]


        self.img = real_img_list+fake_img_list
        self.label = real_label_list+fake_label_list
        
        print('directory, realimg, fakeimg:', self.root, len(real_img_list), len(fake_img_list))


    def __getitem__(self, index):
        img, target = imageio.imread(self.img[index]), self.label[index]
        #print("img file: ", self.img[index])
        imgname = self.img[index]
        #print(imgname)
        if len(img.shape) < 3:
            img=np.asarray(img)[..., np.newaxis]
        if len(img.shape) == 3 and img.shape[-1]==1:
            img=np.tile(np.asarray(img), (1,1,3))
        img = Image.fromarray(img, mode='RGB')

        # compute scaling

        # height, width = img.height, img.width
        # print('height',height )
        # print('width',width)
        height=width=256
        img = data_augment(img, self.opt)

        if self.opt.isTrain and not self.opt.no_flip:
            img = transforms.RandomHorizontalFlip()(img)
        
        input_img = copy.deepcopy(img)
        input_img = transforms.Resize(self.opt.loadSize)(input_img)                #删除这一行和下一行
        input_img = transforms.CenterCrop(self.opt.loadSize)(input_img)
        input_img = transforms.ToTensor()(input_img)
        input_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(input_img)
        # height2, width2 = input_img.height, input_img.width
        img = transforms.Resize(self.opt.cropSize)(img)
        img = transforms.CenterCrop(self.opt.cropSize)(img)
        cropped_img = transforms.ToTensor()(img)
        cropped_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(cropped_img)


        scale = torch.tensor([height, width])
        # scale = torch.tensor([height2, width2])

        return input_img, cropped_img, target, scale, imgname

    def __len__(self):
        return len(self.label) 


