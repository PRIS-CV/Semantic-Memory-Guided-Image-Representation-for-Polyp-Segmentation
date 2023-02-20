import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from lib.UNet_ResNet34 import DecoderBlock
from lib.UNet_ResNet34 import ResNet34Unet
from lib.modules import *

class SeMemory(ResNet34Unet):
    def __init__(self,
                 bank_size =16,
                 decay_rate=0.99,
                 num_classes=1,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=3,
                 pretrained=True,
                 feat_channels=512
                 ):
        super().__init__(num_classes=1,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=3,
                 pretrained=True)
        
        self.feat_channels = feat_channels     
        self.decay_rate = decay_rate
        self.bank_size = bank_size
        self.register_buffer("pixelmemory", torch.zeros(self.bank_size, self.feat_channels))  # pixel semantic memory

        self.l = nn.Conv2d(self.feat_channels, num_classes, 1)
        self.conv_v_x = conv1d(self.feat_channels, self.feat_channels)
        self.conv_memory = conv1d(self.feat_channels, self.feat_channels)
        self.logit_softmax_x = nn.Softmax(dim=1)
        self.logit_softmax_m = nn.Softmax(dim=0)
        self.gamma = Parameter(torch.zeros(1))
        self.memory_weight_init()

        self.conv_l3 =  nn.Sequential(conv2d(self.feat_channels, 256, kernel_size=1),
                                        nn.Upsample(scale_factor=2))
        self.conv_l2 =  nn.Sequential(conv2d(self.feat_channels, 128, kernel_size=1),
                                        nn.Upsample(scale_factor=4))
        self.conv_l1 =  nn.Sequential(conv2d(self.feat_channels, 64, kernel_size=1),
                                        nn.Upsample(scale_factor=8))

        
    #==Initiate the weight of memory==#
    def memory_weight_init(self):
        #nn.init.xavier_normal_(self.memory)
        nn.init.kaiming_uniform_(self.pixelmemory, mode='fan_in')
    
    def memory_att(self, x, flag):
        batch_size, _, height, width = x.shape
        # key = B * C * HW
        key = x.view(batch_size, self.feat_channels, -1) 
        # query = S * C
        query = self.conv_memory(self.pixelmemory.unsqueeze(2)).squeeze(2) 
        # value_memory = S * C
        value = self.conv_memory(self.pixelmemory.unsqueeze(2)).squeeze(2)

        # logit = B * S * HW
        logit = torch.matmul(query, key)
        attn_map = self.logit_softmax_x(logit)

        # feature augmentation
        value = value.transpose(0, 1) # value_memory 
        x_aug = torch.matmul(value, attn_map) # B * C * HW
        x_aug = x_aug.view(batch_size, -1, height, width) # B * C * H * W

        out = self.gamma * x_aug + x
        aux_out = self.l(x)
        # memory update
        if (flag == 'train'):
            x_obj = x * aux_out.sigmoid()
            self.memory_update(x_obj)
        return aux_out, out

    @torch.no_grad()
    def memory_update(self, x_obj):
        # value_x = B * C * HW
        batch_size, _, height, width = x_obj.shape
        value_x = self.conv_v_x(x_obj.view(batch_size, self.feat_channels, -1))
        logit = torch.matmul(self.pixelmemory, value_x)
        attn_map = self.logit_softmax_m(logit) # B * S * HW
        memory_update = attn_map.transpose(0,1).contiguous().view(self.bank_size, -1) @ value_x.transpose(1,2).contiguous().view(-1, self.feat_channels)  #S * BHW @ BHW * C = S * C
        self.pixelmemory = self.decay_rate * self.pixelmemory + (1 - self.decay_rate) * memory_update
       
    def down(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)        
        return e4, e3, e2, e1
    
    def up(self, feat, e3, e2, e1, x):
        feat_l3 = self.conv_l3(feat)
        feat_l2 = self.conv_l2(feat)
        feat_l1 = self.conv_l1(feat)
        center = self.center(feat)     
        d4 = self.decoder4(torch.cat([center, feat_l3+e3], 1))
        d3 = self.decoder3(torch.cat([d4, feat_l2+e2], 1))
        d2 = self.decoder2(torch.cat([d3, feat_l1+e1], 1))
        d1 = self.decoder1(torch.cat([d2, x], 1))
 
        f1 = self.finalconv1(d1)
        f2 = self.finalconv2(d2)
        f3 = self.finalconv3(d3)
        f4 = self.finalconv4(d4)
                
        f4 = F.interpolate(f4, scale_factor=8, mode='bilinear', align_corners=True)
        f3 = F.interpolate(f3, scale_factor=4, mode='bilinear', align_corners=True)
        f2 = F.interpolate(f2, scale_factor=2, mode='bilinear', align_corners=True)
        
        return f4, f3, f2, f1
   
    def forward(self, x, flag):
        batch_size = x.shape[0]
        #=== Stem ===#
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)
        #=== Encoder ===#
        e4, e3, e2, e1  = self.down(x_)        
        #=== Pixel Memory ===#
        aux_out, feats = self.memory_att(e4, flag)
        #=== Decoder ===#
        f4, f3, f2, f1 = self.up(feats, e3, e2, e1, x)
        aux_out = F.interpolate(aux_out, scale_factor=32, mode='bilinear', align_corners=True)

            
        return aux_out, f4, f3, f2, f1