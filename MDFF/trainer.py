import random

import torch
import torch.nn as nn
from networks.resnet import resnet50, resnet502,resnet34
from networks.base_model import BaseModel, init_weights
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from torchvision import transforms
from PIL import Image, ImageDraw
from scipy.fft import fft2, fftshift, ifft2


class SA_layer(nn.Module):
    def __init__(self, dim=128, head_size=4):
        super(SA_layer, self).__init__()
        self.mha = nn.MultiheadAttention(dim, head_size)
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.ac = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        batch_size, len_size, fea_dim = x.shape
        x = torch.transpose(x, 1, 0)
        y, _ = self.mha(x, x, x)
        x = self.ln1(x + y)
        x = torch.transpose(x, 1, 0)
        x = x.reshape(batch_size * len_size, fea_dim)
        x = x + self.fc2(self.ac(self.fc1(x)))
        x = x.reshape(batch_size, len_size, fea_dim)
        x = self.ln2(x)
        return x


class COOI():  # Coordinates On Original Image
    def __init__(self):
        self.stride = 32
        self.cropped_size = 224
        self.score_filter_size_list = [[3, 3], [2, 2]]
        self.score_filter_num_list = [3, 3]
        self.score_nms_size_list = [[3, 3], [3, 3]]
        self.score_nms_padding_list = [[1, 1], [1, 1]]
        self.score_corresponding_patch_size_list = [[224, 224], [112, 112]]
        self.score_filter_type_size = len(self.score_filter_size_list)  # 2

    def get_coordinates(self, fm, scale):  # 获得中心位置后，根据自己设定的patchsize选择方框
        with torch.no_grad():
            batch_size, _, fm_height, fm_width = fm.size()  # fm[batch_size, 2048, 7, 7]
            scale_min = torch.min(scale, axis=1, keepdim=True)[0].long()  # 求每行scale的最小值,一般照片都是[256,256][224,224]?
            scale_base = (scale - scale_min).long() // 2  # torch.div(scale-scale_min,2,rounding_mode='floor'),0
            input_loc_list = []
            for type_no in range(self.score_filter_type_size):  # 2 取0，1循环两次           #先是3,3,后是2,2           #原先疑似是7*7
                score_avg = nn.functional.avg_pool2d(fm, self.score_filter_size_list[type_no],
                                                     stride=1)  # (7,2048,5,5), (7,2048,6,6)
                score_sum = torch.sum(score_avg, dim=1,
                                      keepdim=True)  # (7,1,5,5), (7,1,6,6)把所有的通道数合在一起了,宽度和高度都相加#since the last operation in layer 4 of the resnet50 is relu, thus the score_sum are greater than zero
                _, _, score_height, score_width = score_sum.size()  # 在第一轮是5,第二轮是6
                patch_height, patch_width = self.score_corresponding_patch_size_list[type_no]  # 第一轮224,第二轮112
                # type_no一开始取0，filter_no取0，（0，0），（0，1），（0，2），（1，0），（1，1），（1，2）总共循环六次
                for filter_no in range(self.score_filter_num_list[type_no]):  # 都是3，取0，1，2循环三次
                    score_sum_flat = score_sum.view(batch_size, -1)  # 将四维变成二维,第二维为25或者36
                    value_max, loc_max_flat = torch.max(score_sum_flat, dim=1)  # value指每行的最大值,loc是最大值的位置
                    # loc_max=torch.stack((torch.div(loc_max_flat,score_width,rounding_mode='floor'), loc_max_flat%score_width), dim=1)
                    loc_max = torch.stack((loc_max_flat // score_width, loc_max_flat % score_width),
                                          dim=1)  # 将最大值的横坐标和纵坐标算出来
                    top_patch = nn.functional.max_pool2d(score_sum, self.score_nms_size_list[type_no], stride=1,
                                                         padding=self.score_nms_padding_list[type_no])
                    value_max = value_max.view(-1, 1, 1, 1)
                    erase = (
                                top_patch != value_max).float()  # 如果是不等号，最大的部分每次会被消去，所以会选择不一样的分块，如果是等号，就是三次都给你选一样的区域，这不是我们想看到的
                    score_sum = score_sum * erase  # 它将score_sum经过最大池化后得到的局部最大top_patch和每个通道中最大的value_max作比较,如果一致则置为0,将它从score_sum上消去

                    # location in the original images
                    loc_rate_h = (2 * loc_max[:, 0] + fm_height - score_height + 1) / (2 * fm_height)  # 对每行第一个元素进行操作
                    loc_rate_w = (2 * loc_max[:, 1] + fm_width - score_width + 1) / (2 * fm_width)  # 对每行第二个元素进行操作
                    loc_rate = torch.stack((loc_rate_h, loc_rate_w), dim=1)  # 重新组成矩阵
                    loc_center = (scale_base + scale_min * loc_rate).long()  # loc_rate乘256
                    loc_top = loc_center[:, 0] - patch_height // 2  # 第一列减112或64
                    loc_bot = loc_center[:, 0] + patch_height // 2 + patch_height % 2  # 也是对第一列操作
                    loc_lef = loc_center[:, 1] - patch_width // 2
                    loc_rig = loc_center[:, 1] + patch_width // 2 + patch_width % 2
                    loc_tl = torch.stack((loc_top, loc_lef), dim=1)  # 把loc_center每个元素减去112,会出现负数
                    loc_br = torch.stack((loc_bot, loc_rig), dim=1)  # 把loc_center每个元素加112

                    # For boundary conditions
                    loc_below = loc_tl.detach().clone()  # too low
                    loc_below[loc_below > 0] = 0  # 把loc_tl中大于0的都变0
                    loc_br -= loc_below  # tl变化了多少，br跟着变化，目的是保持二者相差224
                    loc_tl -= loc_below  # 把tl中的元素都变正
                    loc_over = loc_br - scale.long()  # too high，防止br中出现超过256的元素
                    loc_over[loc_over < 0] = 0
                    loc_tl -= loc_over  # tl跟着br变
                    loc_br -= loc_over  # 超过256的变成256     所以最后会有很多的0，224，32，256，这些都是原来不符合标准的，那些不规则的数字才是原先正常的
                    loc_tl[loc_tl < 0] = 0  # patch too large  我感觉tl不会有小于0的数

                    input_loc_list.append(torch.cat((loc_tl, loc_br), dim=1))
                    # print('input_loc_list',input_loc_list)
        input_loc_tensor = torch.stack(input_loc_list, dim=1)  # (7,6,4) 但是我算出来是6，7，4
        # print('input_loc_tensorzong',input_loc_tensor)
        return input_loc_tensor


class Patch5Model(nn.Module):
    def __init__(self, opt):
        super(Patch5Model, self).__init__()
        self.resnet2 = resnet502(pretrained=True)  # debug
        self.resnet = resnet50(pretrained=True)
        self.opt = opt
        self.flag = opt.flag
        self.COOI = COOI()
        self.mha_list = nn.Sequential(
            SA_layer(128, 4),
            SA_layer(128, 4),
            SA_layer(128, 4)
        )
        self.mha_list2 = nn.Sequential(  # 用来融合rgb和ycbcr
            SA_layer(2048, 4),
            SA_layer(2048, 4),
            SA_layer(2048, 4)
        )
        # self.resnet.fc = nn.Linear(2048, 128)
        self.fc1 = nn.Linear(2048, 128)
        self.ac = nn.ReLU()
        self.fc = nn.Linear(128, 1)
        self.avgpool = nn.AvgPool1d(kernel_size=2)

    def forward(self, input_img, cropped_img, scale):

        x = cropped_img
        # y = rgb_to_ycbcr(x)
        # y = calculate_psd(x)
        # x = calculate_psd(x)
        # y = x
        # print(type(x))
        #
        # print('x.shape',x.shape)
        y = rgb_to_ycbcr(x)
        # window_imgs_gpu = y.cuda()
        # window_imgs_cpu = window_imgs_gpu.cpu()
        # window_imgs_cpu = np.clip(window_imgs_cpu, 0, 255)
        # grid = make_grid(window_imgs_cpu, nrow=3)
        # grid = grid.permute(1, 2, 0)
        # y = grid / grid.max()
        #
        # # print('x.shape_gai',x.shape)
        # plt.imshow(y)
        # plt.show()
        # print('trainer',self.opt.flag)
        batch_size, p, _, _ = x.shape  # [batch_size, 3, 224, 224]
        fm, whole_embedding_rgb = self.resnet2(
            x)  # fm[batch_size, 2048, 7, 7], whole_embedding:[batch_size, 2048] 现在给res的就是一张裁减过的图片，后续可以考虑在这里加新特征
        # x = calculate_psd(x)
        _, whole_embedding_ycbcr = self.resnet2(y)
        # fm2, whole_embedding2 = self.resnet2(x)           #这一行以及上一行使得局部分块不使用psd
        # _, whole_embedding_ycbcr = self.resnet2(y)
        # print('whole_embedding.shape',whole_embedding.shape)
        # print('fm.shape',fm.shape)
        whole_embedding_rgb = whole_embedding_rgb.view(-1, 1, 2048)
        whole_embedding_ycbcr = whole_embedding_ycbcr.view(-1, 1, 2048)
        whole_embedding = torch.concat((whole_embedding_rgb, whole_embedding_ycbcr), dim=1)
        whole_embedding = self.mha_list2(whole_embedding)
        whole_embedding = whole_embedding[:, -1]
        # whole_embedding = whole_embedding_ycbcr+whole_embedding_rgb
        # whole_embedding = self.avgpool(whole_embedding.unsqueeze(1)).squeeze(1)
        # fm, whole_embedding=self.resnet2(x)#fm[batch_size, 2048, 7, 7], whole_embedding:[batch_size, 2048] 现在给res的就是一张裁减过的图片，后续可以考虑在这里加新特征
        # print('whole_embedding.shape',whole_embedding.shape)
        # print('fm.shape',fm.shape)
        s_whole_embedding = self.ac(self.fc1(whole_embedding))  # 128，relu[batchsize,128]      我在这里改成了2
        s_whole_embedding = s_whole_embedding.view(-1, 1, 128)  # [batch_size,1,128]
        # print(s_whole_embedding.shape)

        input_loc = self.COOI.get_coordinates(fm.detach(), scale)

        _, proposal_size, _ = input_loc.size()  # 我觉得是6
        # print('input_loc.szie',input_loc.size())
        # print('proposal_size',proposal_size)
        # 我感觉第二个元素是6
        window_imgs = torch.zeros([batch_size, proposal_size, 3, 224, 224]).to(fm.device)  # [N, 4, 3, 224, 224]

        for batch_no in range(batch_size):
            for proposal_no in range(proposal_size):
                t, l, b, r = input_loc[batch_no, proposal_no]
                # print('************************')
                img_patch = input_img[batch_no][:, t:b, l:r]
                # print('t',t)
                # print('b',b)
                # print('l',l)
                # print('r',r)
                # print(img_patch.size())  3，224，224或者3，112，112 取决于你给的框的大小，也可能是64或者32

                # print(img_patch.size())
                _, patch_height, patch_width = img_patch.size()

                ratio = 0.1
                # print('现在掩码率为',ratio)
                patch_size = 8

                # 计算补丁中的像素数量
                num_pixels = patch_height * patch_width

                # 计算要掩盖的像素数量
                num_masked_pixels = int(num_pixels * ratio)

                # 创建掩码图像
                mask = Image.new("L", (patch_width, patch_height), color=255)
                draw = ImageDraw.Draw(mask)

                # 随机选择要掩盖的像素索引
                masked_indices = np.random.choice(num_pixels, num_masked_pixels, replace=False)

                # 根据索引应用掩码
                for index in masked_indices:
                    row_index = index // patch_width
                    col_index = index % patch_width
                    start_x = col_index * patch_size
                    start_y = row_index * patch_size
                    draw.rectangle([start_x, start_y, start_x + patch_size, start_y + patch_size], fill=0)

                # 将掩码图像转换为 NumPy 数组
                mask_np = np.array(mask) / 255.0

                # 将掩码应用于图像补丁
                masked_img_patch = img_patch * mask_np

                _, patch_height, patch_width = masked_img_patch.size()
                if patch_height == 224 and patch_width == 224:
                    window_imgs[batch_no, proposal_no] = masked_img_patch
                else:
                    window_imgs[batch_no, proposal_no:proposal_no + 1] = F.interpolate(masked_img_patch[None, ...],
                                                                                       size=(224, 224),
                                                                                       mode='bilinear',
                                                                                       align_corners=True)  # [N, 6, 3, 224, 224]


        window_imgs = window_imgs.reshape(batch_size * proposal_size, 3, 224, 224)  # [N*6, 3, 224, 224]

        _, window_embeddings = self.resnet(window_imgs.detach())  # [batchsize*self.proposalN, 2048]
        s_window_embedding = self.ac(self.fc1(window_embeddings))  # [batchsize*self.proposalN, 128]
        s_window_embedding = s_window_embedding.view(-1, proposal_size, 128)  # (batchsize,6,128),batchsize目前为7
        # print('s_window_embedding.shape',s_window_embedding.shape)
        # exit()
        # print(window_imgs.shape)

        # window_imgs = calculate_psd(window_imgs)
        # window_imgs_gpu = window_imgs.cuda()
        # window_imgs_cpu = window_imgs_gpu.cpu()
        # window_imgs_cpu = np.clip(window_imgs_cpu, 0, 255)
        # grid = make_grid(window_imgs_cpu, nrow=3)
        # grid = grid.permute(1, 2, 0)
        # grid = grid / grid.max()
        # plt.imshow(grid)
        # plt.axis('off')
        # plt.show()
        all_embeddings = torch.cat((s_window_embedding, s_whole_embedding), 1)  # [1, 1+self.proposalN, 128]
        # all_embeddings=all_embeddings.view(-1, (1+proposal_size), 128)
        # print(all_embeddings.shape)
        all_embeddings = self.mha_list(all_embeddings)  # 三个四头注意力机制
        # print('all_embeddings.shape',all_embeddings.shape)
        all_logits = self.fc(all_embeddings[:, -1])
        # exit()
        # print('all_logits',all_logits)
        return all_logits


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = Patch5Model(opt)
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)

        if not self.isTrain or opt.continue_train:
            # self.model = resnet50(num_classes=1)
            self.model = Patch5Model(opt)
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        if len(opt.gpu_ids) == 0:
            self.model.to('cpu')
        else:
            self.model.to(opt.gpu_ids[0])

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 2.
            if param_group['lr'] < min_lr:
                param_group['lr'] = min_lr
                return False
        return True

    def set_input(self, data):
        self.input_img = data[0]  # (batch_size, 6, 3, 224, 224)
        self.cropped_img = data[1].to(self.device)
        self.label = data[2].to(self.device).float()  # (batch_size)
        self.scale = data[3].to(self.device).float()
        # self.imgname = data[4]

    def forward(self):
        self.output = self.model(self.input_img, self.cropped_img, self.scale)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


def rgb_to_ycbcr(image_batch):
    # 定义转换矩阵
    T = torch.tensor([[0.299, 0.587, 0.114],
                      [-0.168736, -0.331264, 0.5],
                      [0.5, -0.418688, -0.081312]], dtype=torch.float32)

    # 偏移量
    offset = torch.tensor([0, 128, 128], dtype=torch.float32)

    # 确保矩阵和偏移量在相同的设备上
    device = image_batch.device
    T = T.to(device)
    offset = offset.to(device)

    # 获取输入图片的形状
    batch_size, channels, height, width = image_batch.shape
    assert channels == 3, "输入图像的通道数应该为3 (RGB)"

    # 转换为浮点数进行计算
    image_batch = image_batch.float()

    # 调整维度顺序为 (batch_size, height, width, channels)
    image_batch = image_batch.permute(0, 2, 3, 1)

    # 将图像从形状 (batch_size, height, width, channels) 转换为 (batch_size, height*width, channels)
    image_batch = image_batch.reshape((batch_size, -1, 3))

    # 进行矩阵乘法和加法
    ycbcr_batch = torch.matmul(image_batch, T.T) + offset

    # 将结果重新调整为图像的形状 (batch_size, height, width, channels)
    ycbcr_batch = ycbcr_batch.reshape((batch_size, height, width, channels))

    # 调整维度顺序为 (batch_size, channels, height, width)
    ycbcr_batch = ycbcr_batch.permute(0, 3, 1, 2)

    return ycbcr_batch


def calculate_psd(image):
    """
    计算图像的功率谱密度。
    参数:
        image (numpy.ndarray): 形状为 (batch_size, 3, 224, 224) 的图像数组。
    返回:
        numpy.ndarray: 形状为 (batch_size, 3, 224, 224) 的功率谱密度数组。
    """
    # 计算每个颜色通道的傅里叶变换
    if not image.is_cuda:
        image = image.cuda()

        # 将 PyTorch 张量转换为 NumPy 数组
    image_np = image.cpu().numpy()

    # # 计算每个颜色通道的傅里叶变换
    # fft_images = fft2(image_np, axes=(2, 3))
    #
    # # 计算功率谱密度
    # psd_images = np.abs(fft_images) ** 2

    # 应用傅里叶变换
    fft = np.fft.fft2(image_np)
    fft_shift = np.fft.fftshift(fft)  # 将零频率分量移动到频谱中心

    # 计算功率谱密度
    # magnitude_spectrum = 20 * np.log(np.abs(fft_shift))
    magnitude_spectrum = np.abs(fft_shift)

    # 将结果转换回 PyTorch 张量并移回 GPU
    # psd_tensor = torch.from_numpy(psd_images).to(dtype=torch.float32, device=image.device)

    psd_tensor = torch.from_numpy(magnitude_spectrum).to(dtype=torch.float32, device=image.device)

    return psd_tensor

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
