import torch
import torch.nn as nn
from networks.sficresnet import SFIresnet50
from networks.base_model import BaseModel, init_weights
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageDraw

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
        self.score_corresponding_patch_size_list = [[64, 64], [32, 32]]
        self.score_filter_type_size = len(self.score_filter_size_list)  # 2

    def get_coordinates(self, fm, scale):
        with torch.no_grad():
            batch_size, _, fm_height, fm_width = fm.size()
            scale_min = torch.min(scale, axis=1, keepdim=True)[0].long()  # 求每行scale的最小值,一般照片都是[256,256]
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
                                top_patch == value_max).float()  # due to relu operation, the value are greater than 0, thus can be erase by multiply by 1.0/0.0
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
                    loc_br -= loc_below
                    loc_tl -= loc_below
                    loc_over = loc_br - scale.long()  # too high
                    loc_over[loc_over < 0] = 0
                    loc_tl -= loc_over
                    loc_br -= loc_over
                    loc_tl[loc_tl < 0] = 0  # patch too large

                    input_loc_list.append(torch.cat((loc_tl, loc_br), dim=1))

        input_loc_tensor = torch.stack(input_loc_list, dim=1)  # (7,6,4) 但是我算出来是6，7，4
        # print('input_loc_tensor', input_loc_tensor)
        return input_loc_tensor


class Patch5Model(nn.Module):
    def __init__(self):
        super(Patch5Model, self).__init__()
        self.resnet = SFIresnet50(pretrained=True)  # debug
        self.COOI = COOI()
        self.mha_list = nn.Sequential(
            SA_layer(128, 4),
            SA_layer(128, 4),
            SA_layer(128, 4)
        )
        # self.resnet.fc = nn.Linear(2048, 128)
        self.fc1 = nn.Linear(2048, 128)
        self.ac = nn.ReLU()
        self.fc = nn.Linear(128, 1)

    def forward(self, input_img, cropped_img, scale):

        x = cropped_img
        # y = input_img
        # window_imgs_gpu = y.cuda()
        # window_imgs_cpu = window_imgs_gpu.cpu()
        # window_imgs_cpu = np.clip(window_imgs_cpu, 0, 255)
        # grid = make_grid(window_imgs_cpu, nrow=3)
        # grid = grid.permute(1, 2, 0)
        # grid = grid / grid.max()
        # plt.imshow(grid)
        # plt.show()

        batch_size, p, _, _ = x.shape  # [batch_size, 3, 224, 224]

        fm, whole_embedding = self.resnet(x)  # fm[batch_size, 2048, 7, 7], whole_embedding:[batch_size, 2048]
        # print('whole_embedding.shape', whole_embedding.shape)
        # print('fm.shape', fm.shape)
        s_whole_embedding = self.ac(self.fc1(whole_embedding))  # 128，relu
        s_whole_embedding = s_whole_embedding.view(-1, 1, 128)  # [batch_size,1,128]
        # print(s_whole_embedding.shape)

        input_loc = self.COOI.get_coordinates(fm.detach(), scale)

        _, proposal_size, _ = input_loc.size()  # 我觉得是6
        # print('input_loc.szie', input_loc.size())
        # print('proposal_size', proposal_size)
        window_imgs = torch.zeros([batch_size, proposal_size, 3, 224, 224]).to(fm.device)  # [N, 4, 3, 224, 224]

        for batch_no in range(batch_size):
            for proposal_no in range(proposal_size):
                t, l, b, r = input_loc[batch_no, proposal_no]
                # print('************************')
                img_patch = input_img[batch_no][:, t:b, l:r]
                # window_imgs_gpu = img_patch.cuda()
                # window_imgs_cpu = window_imgs_gpu.cpu()
                # window_imgs_cpu = np.clip(window_imgs_cpu, 0, 255)
                # grid = make_grid(window_imgs_cpu, nrow=3)
                # grid = grid.permute(1, 2, 0)
                # grid = grid / grid.max()
                # image_np = grid.cpu().numpy()
                # image_np = (image_np*256).astype(np.uint8)
                # masked_image = Image.fromarray(image_np)
                # plt.imshow(masked_image)
                # plt.show()
                # print('t', t)
                # print('b', b)
                # print('l', l)
                # print('r', r)
                # print(img_patch.size())
                _, patch_height, patch_width = img_patch.size()
                # print('img_patch',img_patch.size())
                #掩码开始******************************************************************
                # ratio = 0.1
                # patch_size = 16
                # while patch_height % patch_size != 0 or patch_width % patch_size != 0:
                #     patch_size -= 1
                # num_patches = (patch_height * patch_width) // (patch_size * patch_size)
                # mask_patches = int(np.ceil(num_patches * ratio))
                # mask = Image.new("L", (patch_width, patch_height), color=255)
                # draw = ImageDraw.Draw(mask)
                # mask_patch_indices = random.sample(range(num_patches), mask_patches)
                #
                # for index in mask_patch_indices:
                #     start_y = (index // (patch_width // patch_size)) * patch_size
                #     start_x = (index % (patch_width // patch_size)) * patch_size
                #     draw.rectangle([start_x, start_y, start_x + patch_size, start_y + patch_size], fill=0)
                # image_np = (image_np).astype(np.uint8)
                # print('image_np.shape',image_np.shape)
                # mask_np = np.array(mask) / 255.0
                # print('mask_np.shape',mask_np.shape)
                # if len(image_np.shape) == 3:
                #     mask_np = np.expand_dims(mask_np, axis=-1)
                #     mask_np = np.repeat(mask_np, image_np.shape[-1], axis=-1)
                # print('mask_np.shape', mask_np.shape)
                # masked_image_np = image_np * mask_np
                # print('masked_image_np',masked_image_np.shape)
                # img_patch = Image.fromarray(np.uint8(masked_image_np))
                # print('img_patch', img_patch.size())
                #掩码结束**************************************************************
                if patch_height == 224 and patch_width == 224:
                    window_imgs[batch_no, proposal_no] = img_patch
                else:
                    window_imgs[batch_no, proposal_no:proposal_no + 1] = F.interpolate(img_patch[None, ...],
                                                                                       size=(224, 224),
                                                                                       mode='bilinear',
                                                                                       align_corners=True)  # [N, 6, 3, 224, 224]


        window_imgs = window_imgs.reshape(batch_size * proposal_size, 3, 224, 224)  # [N*6, 3, 224, 224]
        _, window_embeddings = self.resnet(window_imgs.detach())  # [batchsize*self.proposalN, 2048]
        s_window_embedding = self.ac(self.fc1(window_embeddings))  # [batchsize*self.proposalN, 128]
        s_window_embedding = s_window_embedding.view(-1, proposal_size, 128)
        # print(s_window_embedding.shape)
        # exit()
        # print(window_imgs.shape)

        # window_imgs_gpu = window_imgs.cuda()
        # window_imgs_cpu = window_imgs_gpu.cpu()
        # window_imgs_cpu = np.clip(window_imgs_cpu, 0, 255)
        # grid = make_grid(window_imgs_cpu, nrow=3)
        # grid = grid.permute(1, 2, 0)
        # grid = grid / grid.max()
        # print(grid.shape)
        # plt.imshow(grid)
        # plt.axis('off')
        # plt.show()
        all_embeddings = torch.cat((s_window_embedding, s_whole_embedding), 1)  # [1, 1+self.proposalN, 128]
        # all_embeddings=all_embeddings.view(-1, (1+proposal_size), 128)
        # print(all_embeddings.shape)
        all_embeddings = self.mha_list(all_embeddings)
        # print('all_embeddings.shape', all_embeddings.shape)
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
            self.model = Patch5Model()
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)

        if not self.isTrain or opt.continue_train:
            # self.model = resnet50(num_classes=1)
            self.model = Patch5Model()
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

