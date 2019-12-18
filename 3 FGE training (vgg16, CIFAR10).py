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


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def fge_lr_cycle(c, epoch, it, a_1, a_2):
    t = ((epoch % c) + it) / c
    if t <= 0.5:
        return (1.0 - 2.0 * t) * a_1 + 2.0 * t * a_2
    else:
        return (2.0 - 2.0 * t) * a_2 + (2.0 * t - 1.0) * a_1

def train_epoch(model, data_iter, criterion, optimizer, epoch, c, a_1, a_2):
    model.train()
    losses = []
    for i, batch in zip(tnrange(len(data_iter)), data_iter):
        images, labels = batch
        images, labels = images.cuda(), labels.cuda()
    
        optimizer.zero_grad()
        
        alpha = fge_lr_cycle(c, epoch, (i + 1) / len(data_iter), a_1, a_2)
        for param in optimizer.param_groups:
            param['lr'] = alpha
        
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
    predictions = []
    targets = []
    with torch.no_grad():
        for _, batch in zip(tnrange(len(data_iter)), data_iter):
            images, labels = batch
            images, labels = images.cuda(), labels.cuda()

            out = model(images)
            loss = criterion(out, labels)
            _, preds = torch.max(out, 1)
            predictions.append(F.softmax(out, dim=-1).cpu().numpy())
            targets.append(labels.cpu().numpy())
            acc += torch.sum(preds == labels).type(torch.float)

            losses.append(loss.item())
    return losses, acc / len(data_iter.dataset), np.concatenate(predictions, axis=0), np.concatenate(targets)

def train(model, opt, crit, train_iter, val_iter, num_epochs, n_classes, ens_dir, c, a_1, a_2):
    train_log, val_log = [], []
    val_accs = []
    predictions_ensemble = np.zeros((len(val_iter.dataset), n_classes), dtype=np.float32)
    labels = None
    counter = 0
    
    t1 = time()
    
    for epoch in range(num_epochs):
        tic = time()
        train_loss = train_epoch(model, train_iter, crit, opt, epoch, c, a_1, a_2)
        tac = time()

        tic1 = time()
        val_loss, val_acc, tmppreds, labels = val(model, val_iter, crit)
        tac1 = time()
        
        if (epoch % c) + 1 == c / 2:
            predictions_ensemble += tmppreds
            torch.save(model.state_dict(), ens_dir + '/base_model_{}.pt'.format(counter))
            counter += 1
        
        val_accs.append(val_acc)
        
        train_log.extend(train_loss)

        val_log.append((len(train_iter) * (epoch + 1), np.mean(val_loss)))

        t2 = time()
        #print('EPOCH {}:'.format(epoch))
        #print('Total time from start: {}min {}s'.format((t2 - t1) // 60, (t2 - t1) % 60))
        #print('TRAIN: {}min {}s for epoch, mean loss = {}'.format((tac - tic) // 60, (tac - tic) % 60, np.mean(train_loss)))
        #print('VAL: {}min {}s for epoch, mean loss = {}, acc = {}'.format((tac1 - tic1) // 60, (tac1 - tic1) % 60, np.mean(val_loss), val_acc))
        
    ensemble_acc = np.mean(np.argmax(predictions_ensemble, axis=1) == labels)
    
    return predictions_ensemble, labels, ensemble_acc, model


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


# First model:
vgg16_cifar10_fge = models.vgg16_bn(pretrained=False, **{'num_classes' : 10})
vgg16_cifar10_fge.load_state_dict(torch.load('vgg16_cifar10_32ep_1.pt'))
vgg16_cifar10_fge = vgg16_cifar10_fge.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(vgg16_cifar10_fge.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9)
optimizer.load_state_dict(torch.load('vgg16_cifar10_32ep_1_opt.pt'))

C = 2
dir_name = 'vgg16_cifar10_fge_1'
a_1 = 0.01
a_2 = 0.0005

predictions_ensemble1, test_labels, vgg16_cifar10_fge_1_acc, vgg16_cifar10_fge = train(vgg16_cifar10_fge,
                                                   optimizer, 
                                                   criterion, 
                                                   dataset_loader_train, 
                                                   dataset_loader_test, 
                                                   8, 10, dir_name, C, a_1, a_2)

#print('Ensemble 1 accuracy:', vgg16_cifar10_fge_1_acc)


# Second model:
vgg16_cifar10_fge = models.vgg16_bn(pretrained=False, **{'num_classes' : 10})
vgg16_cifar10_fge.load_state_dict(torch.load('vgg16_cifar10_32ep_2.pt'))
vgg16_cifar10_fge = vgg16_cifar10_fge.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(vgg16_cifar10_fge.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9)
optimizer.load_state_dict(torch.load('vgg16_cifar10_32ep_2_opt.pt'))

C = 2
dir_name = 'vgg16_cifar10_fge_2'
a_1 = 0.01
a_2 = 0.0005

predictions_ensemble2, test_labels, vgg16_cifar10_fge_2_acc, vgg16_cifar10_fge = train(vgg16_cifar10_fge,
                                                   optimizer, 
                                                   criterion, 
                                                   dataset_loader_train, 
                                                   dataset_loader_test, 
                                                   8, 10, dir_name, C, a_1, a_2)

#print('Ensemble 2 accuracy:', vgg16_cifar10_fge_2_acc)


# Third model:
vgg16_cifar10_fge = models.vgg16_bn(pretrained=False, **{'num_classes' : 10})
vgg16_cifar10_fge.load_state_dict(torch.load('vgg16_cifar10_32ep_3.pt'))
vgg16_cifar10_fge = vgg16_cifar10_fge.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(vgg16_cifar10_fge.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9)
optimizer.load_state_dict(torch.load('vgg16_cifar10_32ep_3_opt.pt'))

C = 2
dir_name = 'vgg16_cifar10_fge_3'
a_1 = 0.01
a_2 = 0.0005

predictions_ensemble3, test_labels, vgg16_cifar10_fge_3_acc, vgg16_cifar10_fge = train(vgg16_cifar10_fge,
                                                   optimizer, 
                                                   criterion, 
                                                   dataset_loader_train, 
                                                   dataset_loader_test, 
                                                   8, 10, dir_name, C, a_1, a_2)

#print('Ensemble 3 accuracy:', vgg16_cifar10_fge_3_acc)
