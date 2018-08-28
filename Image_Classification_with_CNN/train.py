import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import utils
import time
import argparse
import os
import pandas as pd
import numpy as np
from PIL import Image
from pdb import set_trace as bp
from guided import GuidedBackprop ,save_gradient_images,save_fooling_images
from fooling import make_fooling_image
from utils import plotNNFilter
from model import FashionMNISTNet,BasicFashionMNISTNet

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=int, default=1, help="default 1 is BEST Model 2 is BASE Model ")
parser.add_argument("--lr", type=float, default=0.001, help="default Learning Rate .001 ")
parser.add_argument("--patience", type=int, default=5, help="early stopping patience")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument("--nepochs", type=int, default=1, help="max epochs")
parser.add_argument("--init", type=int, default=1, help="Custom Initializer : 1 for Xavier, 2 for He")
parser.add_argument("--resume", type=str, default="", help="Saved data Location")
parser.add_argument("--nocuda", action='store_true', help="no cuda used")
parser.add_argument("--aug", type=bool, default=False, help="use augmented data")
parser.add_argument("--nworkers", type=int, default=4, help="number of workers")
parser.add_argument("--seed", type=int, default=1, help="random seed")
args = parser.parse_args()

cuda = not args.nocuda and torch.cuda.is_available() # use cuda
print('Training on cuda: {}'.format(cuda))

# Set seeds. If using numpy this must be seeded too.
if cuda:
    torch.cuda.manual_seed_all(args.seed)

# Setup folders for saved models and logs
if not os.path.exists('saved-models/'):
    os.mkdir('saved-models/')
if not os.path.exists('logs/'):
    os.mkdir('logs/')

# a logs folder for each model and each run.
out_dir = 'logs/{}'.format(args.model)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
run = 0
current_dir = '{}/run-{}'.format(out_dir, run)
while os.path.exists(current_dir):
	run += 1
	current_dir = '{}/run-{}'.format(out_dir, run)
os.mkdir(current_dir)
logfile = open('{}/log.txt'.format(current_dir), 'w')
print(args, file=logfile)

fashion_names=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']


# Define transforms.
train_transforms = transforms.Compose([
                        # transforms.RandomHorizontalFlip(),
                        # utils.RandomRotation(),
                        # utils.RandomTranslation(),
                        # utils.RandomVerticalFlip(),
                        utils.RandomErasing(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ]
                        )
val_transforms = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ])

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if args.init==1:
            nn.init.xavier_uniform(m.weight.data)
        elif args.init==2:
            nn.init.kaiming_uniform(m.weight.data)





###########################################################################################################
##																										 ##
##				Loading The Tranning, Validation &  Testing dataset										 ##
##																										 ##
###########################################################################################################

class TrainDatasetFromCSV():
    def __init__(self, csv_path, height, width, transforms=None):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = pd.read_csv(csv_path)
        self.labels = np.asarray(self.data.iloc[:, 785].values)
        self.height = height
        self.width = width
        self.transforms = transforms
    
    def __getitem__(self, index):
        single_image_label = self.labels[index]

        img_as_np=np.array(self.data.iloc[index,1:785]).reshape(28,28)
    
        img_as_img = Image.fromarray(img_as_np.astype('uint8'))
        img_as_img = img_as_img.convert()
        # Transform image to tensor
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data.index)



class AugmentedDatasetFromCSV():
    def __init__(self, csv_path, height, width, transforms=None):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = pd.read_csv(csv_path)
        self.labels = np.asarray(self.data.iloc[:, 784].values)
        self.height = height
        self.width = width
        self.transforms = transforms
    
    def __getitem__(self, index):
        single_image_label = self.labels[index]

        img_as_np=np.array(self.data.iloc[index,0:784]).reshape(28,28)
    
        img_as_img = Image.fromarray(img_as_np.astype('uint8'))
        img_as_img = img_as_img.convert()
        # Transform image to tensor
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data.index)

class TestDatasetFromCSV():
    def __init__(self, csv_path, height, width, transforms=None):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = pd.read_csv(csv_path)
        self.height = height
        self.width = width
        self.transforms = transforms
    
    def __getitem__(self, index):
        img_as_np=np.array(self.data.iloc[index,1:785]).reshape(28,28)
    

        img_as_img = Image.fromarray(img_as_np.astype('uint8'))
        img_as_img = img_as_img.convert()
        # Transform image to tensor
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        return (img_as_tensor)

    def __len__(self):
        return len(self.data.index)





# Create dataloaders. Use pin memory if cuda.
kwargs = {'pin_memory': True} if cuda else {}


# if args.aug is True augmented data is used to train else Normal Data (as Augmented data creation is too much Time consuming we already generated the augmented data and saved it )
if(args.aug):
    trainset = AugmentedDatasetFromCSV('data/augment_data.csv',28,28,
                            transforms=train_transforms)
else:
    trainset = TrainDatasetFromCSV('data/train.csv',28,28,
                            transforms=train_transforms)


train_loader = DataLoader(trainset, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.nworkers, **kwargs)
valset = TrainDatasetFromCSV('data/val.csv',28,28, transforms=val_transforms)


val_loader = DataLoader(valset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.nworkers, **kwargs)

testset = TestDatasetFromCSV('data/test.csv',28,28, transforms=val_transforms)


test_loader = DataLoader(testset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.nworkers, **kwargs)





###########################################################################################################
##																										 ##
##				Tranning, Validation &  Testing Module										             ##
##																										 ##
###########################################################################################################



def train(net, loader, criterion, optimizer):
    net.train()
    running_loss = 0
    running_accuracy = 0

    for i, (X,y) in enumerate(loader):
        if cuda:
            X, y = X.cuda(), y.cuda()
        X, y = Variable(X), Variable(y)
        # bp()
        output = net(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        running_accuracy += pred.eq(y.data.view_as(pred)).cpu().sum()
    return running_loss/len(loader), running_accuracy/len(loader.dataset)

def validate(net, loader, criterion):
    net.eval()
    running_loss = 0
    running_accuracy = 0
    for i, (X,y) in enumerate(loader):
        if cuda:
            X, y = X.cuda(), y.cuda()
        X, y = Variable(X, volatile=True), Variable(y)
        output = net(X)
        loss = criterion(output, y)
        running_loss += loss.data[0]
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        running_accuracy += pred.eq(y.data.view_as(pred)).cpu().sum()
    return running_loss/len(loader), running_accuracy/len(loader.dataset)


def predict(net, loader ):
    net.eval()
    out_list=[]
    for i, (X) in enumerate(loader):
        if cuda:
            X= X.cuda()
        X= Variable(X, volatile=True)
        output = net(X)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        out_list.append(pred)
    data_out(out_list)

def data_out(out_list):


    pred_list=[]
    for i in out_list :
        for j in i:
            pred_list.append(int(str(j).split("\n")[1].lstrip().rstrip()))

    tag = ['id', 'label']
    ids = np.arange(10000).reshape(10000, 1)
    list_pred=np.array(pred_list).reshape(10000,1)
    data_out = np.concatenate((ids, list_pred), axis=1)
    df_out= pd.DataFrame(data_out, columns=tag)
    df_out.to_csv('pytorch_sub' + '.csv', sep=',', encoding='utf-8', index=False)
    


###########################################################################################################
##																										 ##
##				Main Module                                         									 ##
##																										 ##
###########################################################################################################

def guidedBackProp(model):
    loader = DataLoader(valset, batch_size=1,
                        shuffle=True, num_workers=args.nworkers, **kwargs)
    for i, (X,y) in enumerate(loader):
        if cuda:
            X, y ,model= X.cuda(), y.cuda(),model.cuda()
        X, y = Variable(X,requires_grad=True), y
        GBP = GuidedBackprop(model,  X, y)
        # Get gradients
        guided_grads = GBP.generate_gradients()
        if i==10:
            break
        save_gradient_images(guided_grads, str(i)+'th_image_Guided_BP')
    pass


def fooling(model):
    # setting a Target other than original label
    target_y = np.random.randint(0,10)
    loader = DataLoader(valset, batch_size=10,
                        shuffle=True, num_workers=args.nworkers, **kwargs)
    for i, (X,y) in enumerate(loader):
        if cuda:
            X, y ,model= X.cuda(), y.cuda(),model.cuda()
        if i==10:
            break
        X_tensor = X
        X_fooling = make_fooling_image(X_tensor[i:i + 1], target_y, model)

        scores = model(Variable(X_fooling))
        ori_class=y[i:i + 1].cpu().numpy()[0]
        X_fooling_out=X_fooling.cpu()
        assert bool((ori_class != scores.data.max(1)[1]).cpu().numpy()[0]), 'for {}th input the model is not fooled!'.format(i)
        save_fooling_images(X_fooling_out, str(i) + 'th_image_is_'+str(fashion_names[ori_class])+'_as_starting_image_classified_as'+str(fashion_names[scores.data.max(1)[1].cpu().numpy()[0]]))



if __name__ == '__main__':

    if args.model==1:
        net = FashionMNISTNet()
    elif args.model==2:
        net = BasicFashionMNISTNet()
    net.apply(weights_init)
    print(net)
    criterion = torch.nn.CrossEntropyLoss()

    if cuda:
        net, criterion = net.cuda(), criterion.cuda()
    # early stopping parameters
    patience = args.patience
    best_loss = 1e4
    best_prec1=0

    # Print model to logfile
    print(net, file=logfile)

    # Change optimizer for finetuning
    optimizer = optim.Adam(net.parameters(),lr=args.lr, weight_decay=0.0005)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for e in range(args.nepochs):
        start = time.time()
        train_loss, train_acc = train(net, train_loader,
            criterion, optimizer)
        val_loss, val_acc = validate(net, val_loader, criterion)
        end = time.time()

        # print stats
        stats ="""Epoch: {}\t train loss: {:.3f}, train acc: {:.3f}\t
                val loss: {:.3f}, val acc: {:.3f}\t
                time: {:.1f}s""".format( e, train_loss, train_acc, val_loss,
                val_acc, end-start)
        print(stats)
        print(stats, file=logfile)
        if best_prec1 < val_acc:
            best_prec1=val_acc
            predict(net, test_loader)
        is_best_acc=best_prec1 > val_acc
        #early stopping and save best model
        if val_loss < best_loss:
            best_loss = val_loss
            patience = args.patience
            # print(net.state_dict())
            utils.save_model({
                'epoch': e + 1,
                'best_prec1': best_prec1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best_acc, 'saved-models/{}-run-{}.pth.tar'.format(args.model, run))
        else:
            patience -= 1
            if patience == 0:
                print('Run out of patience!')
                break
    # predict(net, test_loader)

    ###### IN case of Base CONFIG         #########################################################################
    if args.model==2:
        filter=net.state_dict()["features.0.weight"].cpu().numpy()
        plotNNFilter(filter,"filter64")
        guidedBackProp(net)
        fooling(net)
    ##################################################################################################


    

