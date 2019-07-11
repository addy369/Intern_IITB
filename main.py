import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import copy
import torch.utils.data
from data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
from model2 import CNNModel
from dataloader_domain_2 import train2
import numpy as np
from test import test
from torchvision import models
from torch import nn
from tensorboardX import SummaryWriter
writer = SummaryWriter()

model_root = 'models'
cuda = True
cudnn.benchmark = True
lr = 1e-3
batch_size = 10
n_epoch = 50

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

loss_class = torch.nn.CrossEntropyLoss()

def Val(alpha):
    val_loss = 0
    correct = 0
    count=0.0
    count_1=0.0
    running_loss=0.0

   

    for data,target in dataloader_validation:
        if cuda:
            data = data.cuda()
            target = target.cuda()
            #input_img = input_img.cuda()
            #class_label = class_label.cuda()

        out=model_conv(data)
        out=out.view(-1,512*7*7)
        

        


        output,_ = my_net(out,alpha)

        err=loss_class(output,target)
        # running_loss=err+running_loss


        # sum up batch loss
        output=output.view(-1)
        if(output[0]>output[1]):
            output=0
        else:
            output=1
        
        count=count+1;
        if(target==output):
            count_1=count_1+1


            
    # print("accuracy val ")
    accuracy=count_1/count

    return accuracy


def Train(alpha):
    running_loss=0
    val_loss = 0
    correct = 0
    count=0.0
    count_1=0.0
   

    for data,target in dataloader_source_eval:
        if cuda:
            data = data.cuda()
            target = target.cuda()
            #input_img = input_img.cuda()
            #class_label = class_label.cuda()

        out=model_conv(data)
        out=out.view(-1,512*7*7)
        output,_ = my_net(out,alpha)
     
        # sum up batch loss
        output=output.view(-1)
        if(output[0]>output[1]):
            
            output=0
        else:
            output=1

        count=count+1;

        if(target==output):
        
            count_1=count_1+1
        
        else:
            continue


            
    # print("accuracy val ")
    accuracy=count_1/count
    return accuracy
# load data

# img_transform_source = transforms.Compose([
#     transforms.Resize(image_size),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.1307,), std=(0.3081,))
# ])

# img_transform_target = transforms.Compose([
#     transforms.Resize(image_size),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# ])
data_dir = '/home/viraf/Aditya'




dataset_source=train2(os.path.join(data_dir,'train'))

# test_data=train(os.path.join(data_dir,'val'))
dataset_target=train2(os.path.join(data_dir,'train_back2'))




# dataset_source = datasets.MNIST(
#     root='dataset',
#     train=True,
#     transform=img_transform_source,
#     download=True
# )

dataloader_source = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=2*batch_size,
    shuffle=True,
    num_workers=8)
dataloader_source_eval = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=1,
    shuffle=True,
    num_workers=8)

# train_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')

# dataset_target = GetLoader(
#     data_root=os.path.join(target_image_root, 'mnist_m_train'),
#     data_list=train_list,
#     transform=img_transform_target
# )

dataloader_target = torch.utils.data.DataLoader(
    dataset=dataset_target,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8)

dataloader_validation = torch.utils.data.DataLoader(
    dataset=dataset_target,
    batch_size=1,
    shuffle=True,
    num_workers=8)

model_conv = models.vgg16(pretrained=True)
model_conv=model_conv.features
model_conv= nn.Sequential(*list(model_conv.children())[:-1])
model_conv.pool=nn.AdaptiveAvgPool2d(output_size=(7,7))


# model_conv = models.resnet18(pretrained=True)
# model_conv.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1))
# for p in model_conv.parameters():
#     p.requires_grad = True
# model_conv = nn.Sequential(*list(model_conv.children())[:-1])
model_conv=model_conv.cuda()

# load model

my_net = CNNModel()


# setup optimizer

optimizer_1 = optim.Adam(my_net.parameters(), lr=lr,weight_decay=1e-1)
optimizer_2 = optim.Adam(model_conv.parameters(), lr=lr)

loss_class = torch.nn.CrossEntropyLoss()
loss_domain = torch.nn.CrossEntropyLoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True
    best_acc = 0.0

# training


for epoch in range(n_epoch):
    running_loss=0.0
    best_model_wts = copy.deepcopy(my_net.state_dict())
    

    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    i = 0
    while i < len_dataloader:
       


        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = data_source_iter.next()
        s_img, s_label = data_source


        my_net.zero_grad()
        batch_size = len(s_label)

        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.long()

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()           
            domain_label = domain_label.cuda()

        input_img=model_conv(s_img)

        class_output, domain_output = my_net(input_data=input_img, alpha=alpha)
        err_s_label = loss_class(class_output, s_label)
        err_s_domain = loss_domain(domain_output, domain_label)

        # training model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_img)

        domain_label = torch.ones(batch_size)
        domain_label = domain_label.long()

        if cuda:
            t_img = t_img.cuda()
            domain_label = domain_label.cuda()

        input_img=model_conv(t_img)

        _, domain_output = my_net(input_data=input_img, alpha=alpha)
        err_t_domain = loss_domain(domain_output, domain_label)

        err = err_t_domain + err_s_domain + err_s_label
        running_loss=err+running_loss
        err.backward()
        optimizer_1.step()

        i += 1


        print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
              % (epoch, i, len_dataloader, err_s_label.data.cpu().numpy(),
                 err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
    

    writer.add_scalar('data/err',(running_loss/len(dataset_source)),epoch)
                
                
            


    print("training accuracy is")

    train_acc=Train(alpha)
    print(train_acc)
    print("val accuracy is")
    val_acc=Val(alpha)
    print(val_acc)
    print("train loss is")
    print(running_loss/len(dataset_source))
    

    writer.add_scalar('data/accuracy_train',train_acc,epoch)
    writer.add_scalar('data/accuracy_val',val_acc,epoch)

    if(val_acc>best_acc):
        best_acc = val_acc
        
        best_model_wts = copy.deepcopy(my_net.state_dict())



    torch.save(my_net, '{0}/target_domain_{1}.pth'.format(model_root, epoch))

my_net.load_state_dict(best_model_wts)
print("best test accuracy is")

print(best_acc)

