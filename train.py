import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

import timm
from dataset.isic_dataset import SkinDataset

from torchvision import transforms, models
from sklearn.metrics import balanced_accuracy_score

import copy
from torch.utils.data import DataLoader
from optparse import OptionParser
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
import utils
import matplotlib.pyplot as plt
import os
import sys
import time
import math
import pdb

DEBUG = False



dataset_dict = {
    'isic2018': SkinDataset,
}

def train_net(model, config):

    print(config.unique_name)
    
    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    transform = timm.data.create_transform(**data_cfg)
    
    transform_list = [transforms.ToPILImage()]
    transform_list.append(transforms.RandomResizedCrop(size=data_cfg['input_size'][-1], scale=(0.75, 1.0), ratio=(0.75, 1.33), interpolation=utils.get_interpolation_mode(data_cfg['interpolation'])))
    transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.append(transforms.RandomVerticalFlip())
    transform_list.append(transforms.ToTensor())
    
    if config.dataset == 'isic2018':
        transform_list.append(utils.gray_world())
    transform_list.append(transforms.Normalize(mean=data_cfg['mean'], std=data_cfg['std']))

    train_transforms = transforms.Compose(transform_list)
    val_transforms = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(size=int(data_cfg['input_size'][-1]/data_cfg['crop_pct']), interpolation=utils.get_interpolation_mode(data_cfg['interpolation'])),
                        transforms.CenterCrop(size=data_cfg['input_size'][-1]),
                        transforms.ToTensor(),
                        utils.gray_world() if config.dataset=='isic2018' else utils.identity(),
                        transforms.Normalize(mean=data_cfg['mean'], 
                                            std=data_cfg['std'])]
                        )


    trainset = dataset_dict[config.dataset](config.data_path, mode='train', transforms=train_transforms, flag=config.flag, debug=DEBUG, config=config)
    trainLoader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=8, drop_last=True)

    valset = dataset_dict[config.dataset](config.data_path, mode='val', transforms=val_transforms, flag=config.flag, debug=DEBUG, config=config)
    valLoader = DataLoader(valset, batch_size=config.batch_size, shuffle=False, num_workers=2, drop_last=False)
    
    testset = dataset_dict[config.dataset](config.data_path, mode='test', transforms=val_transforms, flag=config.flag, debug=DEBUG, config=config)
    testLoader = DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=2, drop_last=False)

    
    writer = SummaryWriter(config.log_path+config.unique_name)
    
    if config.cls_weight == None:
        criterion = nn.CrossEntropyLoss().cuda() 
    else:
        lesion_weight = torch.FloatTensor(config.cls_weight).cuda()
        criterion = nn.CrossEntropyLoss(weight=lesion_weight).cuda()
    
    if config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=0.0005)
    elif config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    scaler = torch.cuda.amp.GradScaler() if config.amp else None

    BMAC, acc, _ = validation(model, valLoader, criterion)
    print('BMAC: %.5f, Acc: %.5f'%(BMAC, acc))

    best_acc = 0
    for epoch in range(config.epochs):
        print('Starting epoch {}/{}'.format(epoch+1, config.epochs))
        batch_time = 0
        epoch_loss = 0


        model.train()
        
        end = time.time()
        
        exp_scheduler = utils.exp_lr_scheduler_with_warmup(optimizer, init_lr=config.lr, epoch=epoch, warmup_epoch=config.warmup_epoch, max_epoch=config.epochs)

        for i, (data, label) in enumerate(trainLoader, 0):
            x1, target1 = data.float().cuda(), label.long().cuda()
            
            optimizer.zero_grad()

            if config.amp:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    output = model(x1)

                    loss = criterion(output, target1)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            else:
                output = model(x1)

                loss = criterion(output, target1)
                loss.backward()
                optimizer.step()
            

            epoch_loss +=  loss.item()

            batch_time = time.time() - end

            end = time.time()


            print(i, 'loss: %.5f, batch_time: %.5f' % (loss.item(), batch_time))
        
        print('[epoch %d] epoch loss: %.5f' % (epoch+1, epoch_loss/(i+1) ))

        writer.add_scalar('Train/Loss', epoch_loss/(i+1), epoch+1)



        if not os.path.isdir('%s%s/'%(config.cp_path, config.unique_name)):
            os.makedirs('%s%s/'%(config.cp_path, config.unique_name))
        
        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(), '%s%s/CP%d.pth'%(config.cp_path, config.unique_name, epoch+1))

        val_BMAC, val_acc, val_loss = validation(model, valLoader, criterion)
        writer.add_scalar('Val/BMAC', val_BMAC, epoch+1)
        writer.add_scalar('Val/Acc', val_acc, epoch+1)
        writer.add_scalar('Val/val_loss', val_loss, epoch+1)
        
        test_BMAC, test_acc, test_loss = validation(model, testLoader, criterion)
        writer.add_scalar('Test/BMAC', test_BMAC, epoch+1)
        writer.add_scalar('Test/Acc', test_acc, epoch+1)
        writer.add_scalar('Test/test_loss', test_loss, epoch+1)
                
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR/lr', lr, epoch+1)


        if val_BMAC >= best_acc:
            best_acc = val_BMAC
            if not os.path.exists(config.cp_path):
                os.makedirs(config.cp_path)
            torch.save(model.state_dict(), '%s%s/best.pth'%(config.cp_path, config.unique_name))
          

        print('save done')
        print('BMAC: %.5f/best BMAC: %.5f, Acc: %.5f'%(val_BMAC, best_acc, val_acc))


        
def validation(model, dataloader, criterion):
    
    net = model

    net.eval()
    
    losses = 0

    pred_list = np.zeros((0), dtype=np.uint8)
    gt_list = np.zeros((0), dtype=np.uint8)

    with torch.no_grad():
        for i, (data, label) in enumerate(dataloader):
            data, label = data.float(), label.long()

            inputs, labels = data.cuda(), label.cuda()
            pred = net(inputs)

            loss = criterion(pred, labels)
            losses += loss.item()
            
            _, label_pred = torch.max(pred, dim=1)
            
            
            pred_list = np.concatenate((pred_list, label_pred.cpu().numpy().astype(np.uint8)), axis=0)
            gt_list = np.concatenate((gt_list, label.cpu().numpy().astype(np.uint8)), axis=0)
    
    BMAC = balanced_accuracy_score(gt_list, pred_list)
    correct = np.sum(gt_list == pred_list)
    acc = 100 * correct / len(pred_list)

    return BMAC, acc, losses/(i+1)




if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=150, type='int',
            help='number of epochs')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=128,
            type='int', help='batch size')
    parser.add_option('--warmup_epoch', dest='warmup_epoch', default=5, type='int')
    parser.add_option('--optimizer', dest='optimizer', default='sgd', type='str')
    parser.add_option('-l', '--lr', dest='lr', default=0.01, 
            type='float', help='learning rate')
    parser.add_option('-c', '--resume', type='str', dest='load', default=False,
            help='load pretrained model')
    parser.add_option('-p', '--checkpoint-path', type='str', dest='cp_path',
            default='./checkpoint/', help='checkpoint path')
    parser.add_option('-o', '--log-path', type='str', dest='log_path', 
            default='./log/', help='log path')
    parser.add_option('-m', '--model', type='str', dest='model',
            default='resnet50.a1_in1k', help='use which model in [vit_base_patch16_224.orig_in21k, resnet50.a1_in1k]') # We find vit.orig_in21k is better than CLIP weights
    parser.add_option('--linear-probe', dest='linear_probe', action='store_true', help='if use linear probe finetuning')
    parser.add_option('-d', '--dataset', type='str', dest='dataset', 
            default='isic2018', help='name of datasets')
    parser.add_option('-u', '--unique_name', type='str', dest='unique_name',
            default='test', help='name prefix')
    parser.add_option('--flag', type='int', dest='flag', default=2, help='fold for cross-validation')
    parser.add_option('--gpu', type='str', dest='gpu', default='0')
    parser.add_option('--amp', action='store_true', help='if use mixed precision training')

    (config, args) = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    config.log_path = config.log_path + config.dataset + '/'
    config.cp_path = config.cp_path + config.dataset + '/'
    
    print('use model:', config.model)
    
    num_class_dict = {
        'isic2018': 7,
    }

    data_path_dict = {
        'isic2018': '/data/local/yg397/dataset/isic2018/',
    }

    cls_weight_dict = {
        'isic2018': [1, 0.5, 1.2, 1.3, 1, 2, 2],
    }
    
    epoch_dict = {
        'isic2018': 150,
    }

    config.epochs = epoch_dict[config.dataset]
    config.data_path = data_path_dict[config.dataset]
    config.cls_weight = cls_weight_dict[config.dataset]
    config.num_class = num_class_dict[config.dataset]

    
    net = timm.create_model(config.model, pretrained=True, num_classes=config.num_class)
    if config.linear_probe:
        for name, param in net.named_parameters():
            if 'fc' in name and 'resnet' in config.model:
                param.requires_grad = True
            elif 'head' in name and 'vit' in config.model:
                param.requires_grad = True
            else:
                param.requires_grad = False

    print('num of params', sum(p.numel() for p in net.parameters() if p.requires_grad))


    if config.load:
        net.load_state_dict(torch.load(config.load))
        print('Model loaded from {}'.format(config.load))

    net.cuda()
    

    train_net(net, config)

    print('done')
        

