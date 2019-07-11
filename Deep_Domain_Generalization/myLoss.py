import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F


class cus2(torch.nn.Module):

    def __init__(self):
        super(cus2,self).__init__()

    def forward(self, outputs, labels,beta):
        # reshape labels to give a flat vector of length batch_size*seq_len
        labels.cuda()
        outputs.cuda()

        labels = labels.view(-1)

        beta=beta.reshape(-1).float()
        # mask out 'PAD' tokens
        mask = (labels >= 0).float()
        
        # the number of tokens is the sum of elements in mask

        num_tokens=int(torch.sum(mask).item())

        outputs=torch.nn.functional.log_softmax(outputs, dim=1)
        # pick the values corresponding to labels and multiply by mask
        outputs = outputs[range(outputs.shape[0]), labels]*mask
        outputs=beta*outputs

        # cross entropy loss for all non 'PAD' tokens
        return -torch.sum(outputs)/num_tokens
