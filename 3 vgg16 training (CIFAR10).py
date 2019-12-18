import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import random
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, utils, models
from tqdm import tnrange
from time import time


data_transform_train = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

train_cifar10 = datasets.CIFAR10(root='cifar10/train', train=True, transform=data_transform_train, download=True)
test_cifar10 = datasets.CIFAR10(root='cifar10/test', train=False, transform=data_transform_test, download=True)

dataset_loader_train = torch.utils.data.DataLoader(train_cifar10, batch_size=32, shuffle=True, num_workers=4)
dataset_loader_test = torch.utils.data.DataLoader(test_cifar10, batch_size=32, shuffle=False, num_workers=4)


def train_epoch(model, data_iter, criterion, optimizer):
    model.train()
    losses = []
    for _, batch in zip(tnrange(len(data_iter)), data_iter):
        images, labels = batch
        images, labels = images.cuda(), labels.cuda()
    
        optimizer.zero_grad()
        
        out = model(images)
        loss = criterion(out, labels)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    return losses

def val(model, data_iter, criterion):
    model.eval()
    losses = []
    acc = 0.0
    with torch.no_grad():
        for _, batch in zip(tnrange(len(data_iter)), data_iter):
            images, labels = batch
            images, labels = images.cuda(), labels.cuda()

            out = model(images)
            loss = criterion(out, labels)
            _, preds = torch.max(out, 1)
            acc += torch.sum(preds == labels).type(torch.float)

            losses.append(loss.item())
    return losses, acc / len(data_iter.dataset)

def train(model, opt, crit, train_iter, val_iter, num_epochs, sched, checkpoint_epoch, n_model, n_classes):
    train_log, val_log = [], []
    val_accs = []
    last_acc = 0.0
    
    t1 = time()
    
    for epoch in range(num_epochs):
        tic = time()
        train_loss = train_epoch(model, train_iter, crit, opt)
        tac = time()

        tic1 = time()
        val_loss, val_acc = val(model, val_iter, crit)
        tac1 = time()
        
        val_accs.append(val_acc)
        
        sched.step()
        
        train_log.extend(train_loss)

        val_log.append((len(train_iter) * (epoch + 1), np.mean(val_loss)))

        t2 = time()
        #print('EPOCH {}:'.format(epoch))
        #print('Total time from start: {}min {}s'.format((t2 - t1) // 60, (t2 - t1) % 60))
        #print('TRAIN: {}min {}s for epoch, mean loss = {}'.format((tac - tic) // 60, (tac - tic) % 60, np.mean(train_loss)))
        #print('VAL: {}min {}s for epoch, mean loss = {}, acc = {}'.format((tac1 - tic1) // 60, (tac1 - tic1) % 60, np.mean(val_loss), val_acc))
        last_acc = val_acc
        if epoch == checkpoint_epoch:
            torch.save(model.state_dict(), 'vgg16_cifar{}_32ep_{}.pt'.format(n_classes, n_model))
            torch.save(opt.state_dict(), 'vgg16_cifar{}_32ep_{}_opt.pt'.format(n_classes, n_model))
    
    return train_log, val_log, last_acc, model


# First model:
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

vgg16_cifar10 = models.vgg16_bn(pretrained=False, **{'num_classes' : 10})
vgg16_cifar10 = vgg16_cifar10.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(vgg16_cifar10.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9)

sched = MultiStepLR(optimizer, milestones=[20,30], gamma=0.1)

train_log, val_log, vgg16_cifar10_1_acc, vgg16_cifar10 = train(vgg16_cifar10, optimizer, criterion, dataset_loader_train, dataset_loader_test, 40, sched, 31, 1, 10)

torch.save(vgg16_cifar10.state_dict(), 'vgg16_cifar10_40ep_1.pt')


# Second model:
random.seed(8)
np.random.seed(8)
torch.manual_seed(8)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(8)

vgg16_cifar10 = models.vgg16_bn(pretrained=False, **{'num_classes' : 10})
vgg16_cifar10 = vgg16_cifar10.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(vgg16_cifar10.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9)

sched = MultiStepLR(optimizer, milestones=[20,30], gamma=0.1)

train_log, val_log, vgg16_cifar10_2_acc, vgg16_cifar10 = train(vgg16_cifar10, optimizer, criterion, dataset_loader_train, dataset_loader_test, 40, sched, 31, 2, 10)

torch.save(vgg16_cifar10.state_dict(), 'vgg16_cifar10_40ep_2.pt')


# Third model:
random.seed(3)
np.random.seed(3)
torch.manual_seed(3)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(3)

vgg16_cifar10 = models.vgg16_bn(pretrained=False, **{'num_classes' : 10})
vgg16_cifar10 = vgg16_cifar10.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(vgg16_cifar10.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9)

sched = MultiStepLR(optimizer, milestones=[20,30], gamma=0.1)

train_log, val_log, vgg16_cifar10_3_acc, vgg16_cifar10 = train(vgg16_cifar10, optimizer, criterion, dataset_loader_train, dataset_loader_test, 40, sched, 31, 3, 10)

torch.save(vgg16_cifar10.state_dict(), 'vgg16_cifar10_40ep_3.pt')
