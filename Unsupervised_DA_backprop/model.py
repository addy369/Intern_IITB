
import torch.nn as nn
import torch
from functions import ReverseLayerF
from torchvision import datasets, models, transforms



class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()


        # model_conv= models.vgg16(pretrained=True)
        # # print(model_conv)
        # model_conv=model_conv.features
        # model_conv= nn.Sequential(*list(model_conv.children())[:-1])
        # model_conv.pool=nn.AdaptiveAvgPool2d(output_size=(7,7))
        # self.main=nn.Sequential()
        # self.main.add_module('main',model_conv)
        # print(self.main)

    


        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(512*7*7, 100))
        self.class_classifier.add_module('c_r', nn.ReLU())
        self.class_classifier.add_module('c_d', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100,2))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        # self.class_classifier.add_module('c_fc3', nn.Linear(100, 2))
        # self
        # self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('c_fc1', nn.Linear(512*7*7, 100))
        self.domain_classifier.add_module('c_r', nn.ReLU())
        self.domain_classifier.add_module('c_d', nn.Dropout())
        self.domain_classifier.add_module('c_fc2', nn.Linear(100,2))
        


        # self.feature = nn.Sequential()
        # self.feature.add_module('f_conv1', nn.Conv2d(3, 32, kernel_size=11,stride=2))#size after this is 195 
       	# self.feature.add_module('f_relu1', nn.ReLU(True))
        # self.feature.add_module('f_bn1', nn.BatchNorm2d(32))        
        # self.feature.add_module('f_drop1', nn.Dropout2d())

        # self.feature.add_module('f_conv2', nn.Conv2d(32, 64, kernel_size=5,stride=2))#size is 96
        # self.feature.add_module('f_relu2', nn.ReLU(True))
        # self.feature.add_module('f_bn2', nn.BatchNorm2d(64))
        # self.feature.add_module('f_drop2', nn.Dropout2d())
        # self.feature.add_module('f_pool1', nn.AvgPool2d(2))#size is 48

        # self.feature.add_module('f_conv3', nn.Conv2d(64, 128, kernel_size=3,padding=1))#size is 48
        # self.feature.add_module('f_relu3', nn.ReLU(True))
        # self.feature.add_module('f_bn3', nn.BatchNorm2d(128))
        # self.feature.add_module('f_pool2', nn.AvgPool2d(2))#size is 24        
        
        # self.feature.add_module('f_conv4', nn.Conv2d(128, 256, kernel_size=3,padding=1))
        # self.feature.add_module('f_relu4', nn.ReLU(True))
        # self.feature.add_module('f_bn4', nn.BatchNorm2d(256))
        # self.feature.add_module('f_pool3', nn.AvgPool2d(2))#size is 12

        # self.feature.add_module('f_conv5', nn.Conv2d(256, 512, kernel_size=5))#size is 8
        # self.feature.add_module('f_relu5', nn.ReLU(True))
        # self.feature.add_module('f_drop3', nn.Dropout2d())

        # self.feature.add_module('f_pool4', nn.AvgPool2d(2)) #size is 4
        # self.feature.add_module('f_relu2', nn.ReLU(True))


        
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        # self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        # self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))

        # self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))
    
    def forward(self, input_data, alpha):
        # input_data = input_data.expand(input_data.data.shape[0], 3 ,400, 400)
        # input_data=self.main(input_data)
        # # print(feature.shape)
        
        input_data = input_data.view(-1, 512*7*7)

        
        # # print(feature.shape)
        reverse_feature = ReverseLayerF.apply(input_data, alpha)
        # print(reverse_feature.shape)
        class_output = self.class_classifier(input_data)
        # print(class_output.shape)
        domain_output = self.domain_classifier(reverse_feature)
            
        # print(class_output.shape)
        # print(domain_output)
        return class_output, domain_output



net=CNNModel()

x=torch.rand(1,3,400,400)
y=net(x,-1)

