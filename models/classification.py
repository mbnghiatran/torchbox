import torch
import torch.nn as nn
import torchvision.models as models
import ssl
import time

import copy
import math
import torch
import torch.nn as nn

class SPPLayer(nn.Module):
    def __init__(self, scale_list):
        super(SPPLayer, self).__init__()
        self.scale_list = scale_list

    def forward(self, x):
        '''
        x: a tensor vector of previous convolution layer
        scale_list: list contain multi-scale pooling size
        
        returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        '''    
        batch_size, in_channel, in_h, in_w = x.size()
        scale_list = self.scale_list
        for i in range(len(scale_list)):
            h_wid = int(math.ceil(in_h / scale_list[i]))
            w_wid = int(math.ceil(in_w / scale_list[i]))
            h_pad = int((h_wid*scale_list[i] - in_h + 1)/2)
            w_pad = int((w_wid*scale_list[i] - in_w + 1)/2)
            # print(h_wid,w_wid,h_pad, w_pad,i)
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            out = maxpool(x)
            if(i == 0):
                spp = out.view(batch_size, -1)
            else:
                spp = torch.cat((spp, out.view(batch_size, -1)), 1)
        return spp
        
class SPP3DLayer(nn.Module):
    def __init__(self, scale_list):
        super(SPP3DLayer, self).__init__()
        self.scale_list = scale_list

    def forward(self, x):
        '''
        x: a tensor vector of previous convolution layer
        scale_list: list contain multi-scale pooling size
        
        returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        '''    
        batch_size, in_channel, in_h, in_w, in_t = x.size()
        scale_list = self.scale_list
        for i in range(len(scale_list)):
            h_wid = int(math.ceil(in_h / scale_list[i]))
            w_wid = int(math.ceil(in_w / scale_list[i]))
            t_wid = int(math.ceil(in_t / scale_list[i]))
            h_pad = (h_wid*scale_list[i] - in_h + 1)/2
            w_pad = (w_wid*scale_list[i] - in_w + 1)/2
            t_pad = (t_wid*scale_list[i] - in_t + 1)/2
            maxpool = nn.MaxPool3d((h_wid, w_wid, t_wid), stride=(h_wid, w_wid, t_wid), padding=(h_pad, w_pad, t_pad))
            out = maxpool(x)
            if(i == 0):
                spp = out.view(batch_size, -1)
            else:
                spp = torch.cat((spp, out.view(batch_size, -1)), 1)
        return spp


class Lecnet(nn.Module):
    # here present the code coding for the Lecnet model in the paper
    def __init__(self,class_num=3,num_of_blocks=9,training=True,dense_layers=[256,256]):
        super(Lecnet, self).__init__()
        self.num_blocks = num_of_blocks
        self.class_num = class_num
        self.training = training
        self.model = nn.Sequential()

        for block_num in range(0,self.num_blocks):
            # add cnn blocks
            if (block_num==0):
                in_channels = 1
            else:
                in_channels = 128
            out_channels = 128
            self.model.add_module('block_{}_cnn'.format(block_num),self.depthblock(block_num,in_channels,out_channels))
        ft=16
        self.model.add_module('avg_pool', nn.AdaptiveAvgPool1d(ft))
        self.model.add_module('Flatten', nn.Flatten())
        for index,value in enumerate(dense_layers):
            if (index==0):
                in_dense = ft*out_channels
            else:
                in_dense = dense_layers[index-1]
            # add dense layers
            self.model.add_module('dense_{}'.format(index), 
                                        nn.Linear(in_features = in_dense,
                                                out_features = 256, 
                                                bias=True))
                # # add activation function
                # if (index==(len(dense_layers)-1)):
                #     # add activation after final dense function
            self.model.add_module('dense_activation_{}',
                                nn.Tanh())
            self.model.add_module('dropout  _{}',
                                nn.Dropout(p=0.2))    

        # self.meta_net = nn.Sequential(nn.Linear(1, 64,bias=True),
        #                         #   nn.BatchNorm1d(64),
        #                           nn.Tanh(),
        #                           nn.Dropout(p=0.2),
        #                           nn.Linear(64, 128,bias=True),
                                
        #                           nn.Tanh(),
        #                           nn.Dropout(p=0.2))
        
        self.meta_net = copy.deepcopy(self.model)

        self.out_1 =  nn.Linear(256+256, 128,bias=True)
        self.out_2 = nn.Linear(128,self.class_num,bias=True)

    def depthblock(self,block_index,in_channels,out_channels):
        block = nn.Sequential()
        # block.add_module('padding_{}'.format(block_index),\
        #             nn.ConstantPad1d((8*block_index,8*block_index),0))
        block.add_module('conv_{}_1,1'.format(block_index),\
                    nn.Conv1d(in_channels = in_channels, \
                              out_channels = in_channels, \
                              kernel_size = 3, \
                              stride= 1, \
                              padding= 1, \
                              dilation=1, \
                              groups=in_channels, \
                              bias=True, \
                              padding_mode='zeros'))

        block.add_module('relu_{}_1'.format(block_index),\
                        nn.ReLU())
        block.add_module('BatchNorm_{}_1,1'.format(block_index), \
                         nn.BatchNorm1d(in_channels))                
        # if self.train_:
        #   block.add_module('dropout-{}-1'.format(block_index),\
        #                   nn.Dropout(p = 0.05))
        block.add_module('conv_{}_1,2'.format(block_index),\
                    nn.Conv1d(in_channels = in_channels, \
                              out_channels = out_channels, \
                              kernel_size = 1, \
                              stride= 1, \
                              padding= 1, \
                              dilation=1, \
                              groups=1, \
                              bias=True, \
                              padding_mode='zeros'))

        block.add_module('relu_{}_2'.format(block_index),\
                        nn.ReLU())
        block.add_module('BatchNorm_{}_1,2'.format(block_index),\
                         nn.BatchNorm1d(out_channels))                
        pooling_kernel_size = 2
        if pooling_kernel_size>1:
          block.add_module('maxpool_{}'.format(block_index),\
                          nn.MaxPool1d(kernel_size = 2, \
                                    stride=2, \
                                    padding= 1, \
                                    dilation=1, \
                                    return_indices=False, \
                                    ceil_mode=False))
        if self.training:
          block.add_module('dropout_{}_2'.format(block_index),\
                          nn.Dropout(p = 0.2))

        return block
    
    def forward(self, data, meta):
        # batch_size = data.shape[0]
        batch_size = 1
        # data = data.view(1,1,3850)
        # print(meta.view(batch_size,1).shape)
        # print(data.shape)
        # time.sleep(5)
        features = self.model(data)
        # print(features.shape)
        features_meta = self.meta_net(meta)

        # features_meta = self.meta_net(meta.view(batch_size,1))
        # print(features.shape)
        # print(features_meta.shape)
        features_cat = torch.cat((features,features_meta),dim=1)
        output = self.out_1(features_cat)
        # output = self.out_1(features)
        output = self.out_2(output)
        # output = features
        # print(output.shape)
        return output

