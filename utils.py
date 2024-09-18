import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.utils import make_grid, save_image
import pdb 
import os
import scipy.ndimage
import types as Types
import torchvision.transforms as transforms

class gray_world(object):
    
    def __call__(self, img):
        mu_g = img[1].mean()
        img[0] = img[0] * (mu_g / img[0].mean())
        img[2] = img[2] * (mu_g / img[2].mean())

        img = torch.clamp(img, 0, 1)

        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'

class identity(object):
    
    def __call__(self, img):

        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class my_resize(object):

    def __init__(self, size):
        self.size = size
    
    def __call__(self, img):
        
        img = img.unsqueeze(0)
        img = F.interpolate(img, size=self.size, mode='bilinear', align_corners=True)

        img =  img.squeeze(0)
        

        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'

def get_interpolation_mode(interpolation_str):
    interpolation_mapping = {
        'nearest': transforms.InterpolationMode.NEAREST,
        'lanczos': transforms.InterpolationMode.LANCZOS,
        'bilinear': transforms.InterpolationMode.BILINEAR,
        'bicubic': transforms.InterpolationMode.BICUBIC,
        'box': transforms.InterpolationMode.BOX,
        'hamming': transforms.InterpolationMode.HAMMING
    }
    return interpolation_mapping.get(interpolation_str)
 

class GaussianLayer(nn.Module):
    def __init__(self, sigma=8):
        super(GaussianLayer, self).__init__()
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(10),
            nn.Conv2d(1, 1, 21, stride=1, padding=0, bias=None)
            )
        self.weights_init(sigma)
        #self.seq = self.seq.cuda()
  

    def forward(self, x): 
        return self.seq(x)

    def weights_init(self, sigma):
        n = np.zeros((21, 21))
        n[10, 10] = 1 
        k = scipy.ndimage.gaussian_filter(n, sigma=sigma)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))
            f.requires_grad = False



def save_vis_imgs_3(model, imgs_vis, imgs_vis_label, writer, epoch, vis_dir, config):
    
    noise = torch.randn(imgs_vis.size(0), config.noise_dim).cuda()
    imgs_vis_aug = model.aug_net(noise, imgs_vis.cuda(), imgs_vis_label.cuda())
    imgs_vis_aug = imgs_vis_aug.cpu()

    grid = make_grid(imgs_vis_aug[:imgs_vis.size(0)], nrow=int(math.sqrt(imgs_vis.size(0))), normalize=True, padding=1, pad_value=1)
	

    save_image(grid, os.path.join(vis_dir, 'aug_imgs_%d.png'%(epoch+1)))

def save_vis_imgs_4(model, imgs_vis, imgs_vis_label, writer, epoch, vis_dir, config):
    
    noise = torch.randn(imgs_vis.size(0), config.noise_dim).cuda()
    imgs_vis_aug, imgs_vis_label_aug = model.aug_net(noise, imgs_vis.cuda(), imgs_vis_label.cuda())
    imgs_vis_aug = imgs_vis_aug.cpu()
    imgs_vis_label_aug = imgs_vis_label_aug.cpu()

    grid1 = make_grid(imgs_vis_aug[:imgs_vis.size(0)], nrow=int(math.sqrt(imgs_vis.size(0))), normalize=True, padding=1, pad_value=1)
    grid2 = make_grid(imgs_vis_aug[imgs_vis.size(0):], nrow=int(math.sqrt(imgs_vis.size(0))), normalize=True, padding=1, pad_value=1)

    grid = torch.cat([grid1, grid2], dim=2)
	

    save_image(grid, os.path.join(vis_dir, 'aug_imgs_%d.png'%(epoch+1)))

def save_vis_imgs_5(model, imgs_vis, imgs_vis_label, writer, epoch, vis_dir, config):
    
    noise = torch.randn(imgs_vis.size(0), config.noise_dim).cuda()
    imgs_vis_aug, _= model.aug_net(noise, imgs_vis.cuda())
    imgs_vis_aug = imgs_vis_aug.cpu()

    grid = make_grid(imgs_vis_aug[:imgs_vis.size(0)], nrow=int(math.sqrt(imgs_vis.size(0))), normalize=True, padding=1, pad_value=1)
	

    save_image(grid, os.path.join(vis_dir, 'aug_imgs_%d.png'%(epoch+1)))



def multistep_lr_scheduler_with_warmup(optimizer, init_lr, epoch, warmup_epoch, lr_decay_epoch, max_epoch, gamma=0.1):

    if epoch >= 0 and epoch <= warmup_epoch:
        lr = init_lr * 2.718 ** (10*(float(epoch) / float(warmup_epoch) - 1.))
        if epoch == warmup_epoch:
            lr = init_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    flag = False
    for i in range(len(lr_decay_epoch)):
        if epoch == lr_decay_epoch[i]:
            flag = True
            break

    if flag == True:
        lr = init_lr * gamma**(i+1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    else:
        return optimizer.param_groups[0]['lr']

    return lr

def exp_lr_scheduler_with_warmup(optimizer, init_lr, epoch, warmup_epoch, max_epoch):

    if epoch >= 0 and epoch <= warmup_epoch:
        lr = init_lr * 2.718 ** (10*(float(epoch) / float(warmup_epoch) - 1.))
        if epoch == warmup_epoch:
            lr = init_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    else:
        lr = init_lr * (1 - epoch / max_epoch)**0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return lr

class Exp_LR_Scheduler_with_Warmup():
    def __init__(self, optimizer, init_lr, warmup_epoch, max_epoch):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.max_epoch = max_epoch
        self.warmup_epoch = warmup_epoch
        self.current_epoch = 0
        
        lr = self.init_lr * 2.718 ** (10*(float(self.current_epoch) / float(warmup_epoch) - 1.))
        self.set_lr(lr)

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def step(self):
        
        self.current_epoch += 1

        if self.current_epoch >= 0 and self.current_epoch <= self.warmup_epoch:
            lr = self.init_lr * 2.718 ** (10*(float(self.current_epoch) / float(self.warmup_epoch) - 1.))
            if self.current_epoch == self.warmup_epoch:
                lr = self.init_lr
        else:
            lr = self.init_lr * (1 - self.current_epoch / self.max_epoch) ** 0.9

        self.set_lr(lr)

class MultiBatchNorm(nn.Module):
    def __init__(self, num_features, momentum=None, eps=1e-05, dim='2d', types=['base', 'aug']) :
        assert isinstance(types, list) and len(types) > 1 
        assert 'base' in types
        assert dim in ('1d', '2d')
        super(MultiBatchNorm, self).__init__()
        self.types = types

        if dim == '1d':
            if momentum is not None:
                self.bns = nn.ModuleDict([[t, nn.BatchNorm1d(num_features, momentum=momentum, eps=eps)] for t in types]) 
            else:
                self.bns = nn.ModuleDict([[t, nn.BatchNorm1d(num_features, eps=eps)] for t in types])        
        elif dim == '2d':
            if momentum is not None:
                self.bns = nn.ModuleDict([[t, nn.BatchNorm2d(num_features, momentum=momentum, eps=eps)] for t in types])
            else:
                self.bns = nn.ModuleDict([[t, nn.BatchNorm2d(num_features, eps=eps)] for t in types])

        self.t = 'base'
    def forward(self, x):
        # print('bn type: {}'.format(self.t))
        assert self.t in self.types
        out = self.bns[self.t](x)
        self.t = 'base'
        return out 
def replace_bn_with_multibn(model, types=['base', 'aug']):
    def convert(model):
        conversion_count = 0
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                    
                model._modules[name], num_converted = convert(module)
                conversion_count += num_converted

            if type(module) == nn.BatchNorm2d:
            
                layer_old = module
                num_features = module.num_features
                eps = module.eps
                momentum = module.momentum
                layer_new = MultiBatchNorm(num_features=num_features, eps=eps, momentum=momentum, types=types)

                state_dict = module.state_dict()
                for t in types:
                    layer_new.bns[t].load_state_dict(state_dict)

                model._modules[name] = layer_new
                conversion_count += 1
        return model, conversion_count

   
    def set_bn_type(self, t):
        for m in self.modules():
            if isinstance(m, MultiBatchNorm):
                m.t = t

    model, _ = convert(model)
    model.set_bn_type = Types.MethodType(set_bn_type, model)

    return model

def replace_bn_with_layer_norm(model, types='layer'):
    def convert_bn(model):
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                
                model._modules[name] = convert_bn(module)
            if type(module) == nn.BatchNorm2d:
                
                layer_old = module
                num_features = module.num_features
                eps = module.eps
                momentum = module.momentum
                
                if types == 'no':
                    layer_new = nn.Sequential(nn.Identity())
                model._modules[name] = layer_new
        return model
    model = convert_bn(model)

    return model


def cal_dice(pred, target, C): 
    N = pred.shape[0]
    target_mask = target.data.new(N, C).fill_(0)
    target_mask.scatter_(1, target, 1.) 

    pred_mask = pred.data.new(N, C).fill_(0)
    pred_mask.scatter_(1, pred, 1.) 

    intersection= pred_mask * target_mask
    summ = pred_mask + target_mask

    intersection = intersection.sum(0).type(torch.float32)
    summ = summ.sum(0).type(torch.float32)
    
    eps = torch.rand(C, dtype=torch.float32)
    eps = eps.fill_(1e-7)

    summ += eps.cuda()
    dice = 2 * intersection / summ

    return dice, intersection, summ

