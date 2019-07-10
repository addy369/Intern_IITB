from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter
from dataloader_domain import train
from dataloader_domain_2 import train2
from model2 import CNNModel
from myLoss import cus2

writer = SummaryWriter()



data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor()
        
    ]),
    'val': transforms.Compose([transforms.ToTensor(),
    ]),
}

data_dir = '/home/viraf/Aditya'




source_data=train(os.path.join(data_dir,'train'))

# test_data=train(os.path.join(data_dir,'val'))
test_data=train2(os.path.join(data_dir,'test'))

validation_split = 0.2

dataset_size = len(test_data)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

np.random.shuffle(indices)
valid_indices, test_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
valid_sampler = SubsetRandomSampler(valid_indices)
test_sampler = SubsetRandomSampler(test_indices)

dataloader_train=torch.utils.data.DataLoader(source_data,batch_size=50,
                                             shuffle=True, num_workers=4)
dataloader_valid=torch.utils.data.DataLoader(test_data,batch_size=50,sampler=valid_sampler)


dataloader_test=torch.utils.data.DataLoader(test_data,batch_size=1,sampler=test_sampler)

# image_dataset_test=datasets.ImageFolder(os.path.join(data_dir,'test'),data_transforms['val'])

dataset_sizes = {'train':len(source_data),'val':len(test_data),'test':len(dataloader_test)}


dataloaders = {'train':dataloader_train,'val':dataloader_valid}

# class_names = source_data.classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset_size = len(source_data)
# indices = list(range(dataset_size))
# split = int(np.floor(validation_split * dataset_size))
# np.random.seed(manual_seed)
# np.random.shuffle(indices)
# train_indices, val_indices = indices[split:], indices[:split]

# # Creating PT data samplers and loaders:
# train_sampler = SubsetRandomSampler(train_indices)
# valid_sampler = SubsetRandomSampler(val_indices)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
# inputs, classes = next(iter(dataloaders['train']))

# # Make a grid from batch
# out = torchvision.utils.make_grid(inputs)

# imshow(out, title=[class_names[x] for x in classes])


def train_model(model, criterion_1,criterion_2,optimizer, scheduler, num_epochs=25,alpha=-0.5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
        
            for inputs,labels,domains,beta in dataloaders[phase]:

                inputs=inputs.cuda()

                
                # inputs = inputs.to(device)
                labels = labels.cuda()
                domains=domains.cuda()
                beta=beta.cuda()

                inputs=model_conv(inputs)
                
                inputs=inputs.cuda()
               
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                # with torch.set_grad_enabled(phase == 'train'):
                #     outputs = model(inputs,alpha,train,labels)
                #     _, preds = torch.max(outputs, 1)
                #     loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        output_r_label=torch.zeros(0,0)
                        output_r_label=output_r_label.cuda()
                        domain_r_output=torch.zeros(0,0)
                        domain_r_output=domain_r_output.cuda()
                        domain_r_output_class=torch.zeros(0,0)
                        domain_r_output_class=domain_r_output_class.cuda()
                        for i in range(0,inputs.shape[0]):
                            x=inputs[i]

                            x=x.unsqueeze(0)
                            x=x.cuda()
                           

                            output_label,domain_output,domain_output_class = model(x,alpha,phase,labels[i])
                           
                            

                            output_r_label=torch.cat((output_r_label,output_label),0)
                            domain_r_output=torch.cat((domain_r_output,domain_output),0)
                            domain_r_output_class=torch.cat((domain_r_output_class,domain_output_class),0)

                            

                        output_label=output_r_label
                        domain_output=domain_r_output
                        domain_output_class=domain_r_output_class
                        output_label.cuda()
                        domain_output.cuda()
                        domain_output_class.cuda()
                        _, preds = torch.max(output_label, 1)


                        loss_label=criterion_1(output_label,labels)
                        loss_domain=criterion_2(domain_output,domains)
                        loss_domain_class=criterion_1(domain_output_class,domains)
                        loss=loss_label+loss_domain+loss_domain_class
                        loss.backward()
                        optimizer.step()

                    if(phase=='val'):
                        outputs=model(inputs,alpha,phase,labels)
                    
                        _, preds = torch.max(outputs, 1)
                        loss = criterion_1(outputs, labels)
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double()/ dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # if(phase=='train'):

            #     writer.add_scalar('data/err_train',epoch_loss,epoch)
            #     writer.add_scalar('data/accuracy_train',epoch_acc,epoch)
            # if(phase=='val'):
                
            #     writer.add_scalar('data/err_val',epoch_loss,epoch)
            #     writer.add_scalar('data/accuracy_val',epoch_acc,epoch)
            


            

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels,img,beta) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs,-1,'val',labels)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)



# model_ft = models.resnet18(pretrained=True)
# model_ft.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1))
# # num_ftrs = model_ft.fc.in_features
# # model_ft.fc = nn.Sequential(
# #                       nn.Linear(25088, 4), 
# #                       nn.Dropout(0.5),
# #                       nn.Linear(4, 2),                   
# #                   )
# model_ft.fc=nn.Linear(512,2)


# model_ft = model_ft.to(device)
model_conv = models.resnet18(pretrained=True)
model_conv.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1))
for p in model_conv.parameters():
    p.requires_grad = True
model_conv = nn.Sequential(*list(model_conv.children())[:-1])
model_conv=model_conv.cuda()



criterion_1 = nn.CrossEntropyLoss()
criterion_2=nn.CrossEntropyLoss()

my_model=CNNModel()
my_model=my_model.cuda()
# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(my_model.parameters(), lr=0.1,weight_decay=1e-3)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


model_ft = train_model(my_model,criterion_1,criterion_2,optimizer_ft, exp_lr_scheduler,num_epochs=25,alpha=-1)

# visualize_model(model_ft)



def test():
    running_loss = 0.0
    running_corrects = 0
    for inputs,labels,domain,beta in dataloader_test:
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                phase='val'
                # zero the parameter gradients
                model_ft.eval()

                # forward
                # track history if only in train
                
                inputs = model_conv(inputs)
                inputs.cuda()

                outputs=model_ft(inputs,-1,'val',labels)

                _, preds = torch.max(outputs, 1)
                
                loss = criterion_1(outputs,labels)
                

                    # backward + optimize only if in training phase
                    
 
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
    print("accuracy is ")
    print(int(running_corrects)/len(dataloader_test))
    



    # print("loss is ")
    # print(running_loss) 
    # print("accuracy is ")
    # print(running_corrects/len(dataloader_test))

test()




