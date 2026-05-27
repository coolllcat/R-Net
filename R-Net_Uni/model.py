import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import warp3d


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, 3, strides, 1),
            nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, base_dim = 32, levels = 4):
        super().__init__()
        self.levels = levels
        self.conv_in = ConvBlock(2, base_dim)
        self.encoder = nn.ModuleList()
        for i in range(levels):
            self.encoder.append(ConvBlock(base_dim * (i + 1), base_dim * (i + 2), 2))

    def forward(self, x_1, x_2):
        x_in = torch.cat([x_1, x_2], dim=1)
        x0 = self.conv_in(x_in)
        encoder_results = [x0]
        for i in range(self.levels):
            encoder_results.append(self.encoder[i](encoder_results[-1]))

        return encoder_results


class Decoder(nn.Module):
    def __init__(self, base_dim = 32, levels = 4):
        super().__init__()
        self.levels = levels
        self.decoder = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        for i in range(levels):
            self.decoder.append(ConvBlock(base_dim * (i + 1 + i + 1), base_dim * (i + 1)))
            self.upconvs.append(nn.ConvTranspose3d(base_dim * (i + 2), base_dim * (i + 1), 3, 2, 1, 1))

    def forward(self, encoder_results):
        x = encoder_results[-1]
        for i in range(self.levels - 1, -1, -1):
            x = self.upconvs[i](x)
            x = self.decoder[i](torch.cat([x, encoder_results[i]], dim=1))
        
        return x
            

class DeepReg(nn.Module):
    def __init__(self, base_dim = 64, levels = 4, cascade = 3):
        super().__init__()
        self.base_dim = base_dim
        self.levels = levels
        self.cascade = cascade
        self.encoder = Encoder(base_dim, levels)
        self.decoder = Decoder(base_dim, levels)
        self.flowout = nn.ModuleList([ConvBlock(base_dim, 3) for i in range(cascade)]) 
            
    def forward(self, x_1, x_2):
        encoder_result = self.encoder(x_1, x_2)
        decoder_result = self.decoder(encoder_result)
        flowresults = []
        for i in range(self.cascade):
            decoder_flow = self.flowout[i](decoder_result)
            if i == 0:
                flowresults.append(decoder_flow)
            else:
                flowresults.append(warp3d(flowresults[-1], decoder_flow) + decoder_flow)

        return flowresults 


class Step_model(nn.Module):
    def __init__(self, base_model, steps = 16, field = 3):
        super().__init__()
        self.base_model = base_model
        self.steps = steps
        self.field = field
        self.stephead = nn.Conv3d(base_model.base_dim, steps * field * 3, 1, 1, 0)

    def forward(self, x_1, x_2):
        with torch.no_grad():
            encoder_result = self.base_model.encoder(x_1, x_2)
            decoder_result = self.base_model.decoder(encoder_result)
        flow_feas = self.stephead(decoder_result)
        flow_feas = flow_feas.reshape(x_1.shape[0], self.steps, self.field, 3, *x_1.shape[2:])
        flow_feas = F.softmax(flow_feas / 0.1, dim=2)

        flow_feas = flow_feas.to("cuda:1")
        self.flow_xyz = torch.tensor([0,-1,1]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to("cuda:1")
        flowresults = []
        for i in range(self.steps):
            flow_fea = flow_feas[:, i, :, :, :, :, :]
            flow_step = torch.sum(flow_fea * self.flow_xyz, dim=1)
            if i == 0:
                flowresults.append(flow_step)
            else:
                flowresults.append(warp3d(flowresults[-1], flow_step) + flow_step)

        return flowresults


class Step_model_full(nn.Module):
    def __init__(self, base_dim = 64, steps = 20, field = 3):
        super().__init__()
        self.base_model = DeepReg(base_dim = base_dim)
        self.base_dim = base_dim
        self.steps = steps
        self.field = field
        self.stephead = nn.Conv3d(base_dim, steps * field * 3, 1, 1, 0)

    def forward(self, x_1, x_2):
        encoder_result = self.base_model.encoder(x_1, x_2)
        decoder_result = self.base_model.decoder(encoder_result)
        flow_feas = self.stephead(decoder_result)
        
        flow_feas = flow_feas.reshape(x_1.shape[0], self.steps, self.field, 3, *x_1.shape[2:])
        flow_feas = F.softmax(flow_feas / 0.1, dim=2)

        self.flow_xyz = torch.tensor([0,-1,1]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
        flowresults = []
        for i in range(self.steps):
            flow_fea = flow_feas[:, i, :, :, :, :, :]
            flow_step = torch.sum(flow_fea * self.flow_xyz, dim=1)
            if i == 0:
                flowresults.append(flow_step)
            else:
                flowresults.append(warp3d(flowresults[-1], flow_step) + flow_step)

        return flowresults


class Step_model_plus(nn.Module):
    def __init__(self, base_dim = 64, steps = 20, field = 3, basemodels = None):
        super().__init__()
        self.base_models = basemodels
        self.base_dim = base_dim
        self.steps = steps
        self.field = field
        self.stephead = nn.Conv3d(base_dim*len(basemodels), steps * field * 3, 1, 1, 0)

    def forward(self, x_1, x_2):
        with torch.no_grad():
            features = []
            for base_model in self.base_models:
                encoder_result = base_model.encoder(x_1, x_2)
                decoder_result = base_model.decoder(encoder_result)
                features.append(decoder_result)
            features = torch.cat(features, dim=1)

        flow_feas = self.stephead(features)
        flow_feas = flow_feas.reshape(x_1.shape[0], self.steps, self.field, 3, *x_1.shape[2:])
        flow_feas = F.softmax(flow_feas / 0.1, dim=2)

        self.flow_xyz = torch.tensor([0,-1,1]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
        flowresults = []
        for i in range(self.steps):
            flow_fea = flow_feas[:, i, :, :, :, :, :]
            flow_step = torch.sum(flow_fea * self.flow_xyz, dim=1)
            if i == 0:
                flowresults.append(flow_step)
            else:
                flowresults.append(warp3d(flowresults[-1], flow_step) + flow_step)

        return flowresults
    

class Step_model_plus_plus(nn.Module):
    def __init__(self, base_dim = 64, steps = 20, field = 3, basemodels = None, stephead = None):
        super().__init__()
        self.base_model1 = basemodels[0]
        self.base_model2 = basemodels[1]
        self.base_model3 = basemodels[2]
        self.base_dim = base_dim
        self.steps = steps
        self.field = field
        self.stephead = stephead

    def forward(self, x_1, x_2):
        encoder_result1 = self.base_model1.encoder(x_1, x_2)
        decoder_result1 = self.base_model1.decoder(encoder_result1)
        encoder_result2 = self.base_model2.encoder(x_1, x_2)
        decoder_result2 = self.base_model2.decoder(encoder_result2)
        encoder_result3 = self.base_model3.encoder(x_1, x_2)
        decoder_result3 = self.base_model3.decoder(encoder_result3)
        features = torch.cat([decoder_result1, decoder_result2, decoder_result3], dim=1)

        flow_feas = self.stephead(features)
        flow_feas = flow_feas.reshape(x_1.shape[0], self.steps, self.field, 3, *x_1.shape[2:])
        flow_feas = F.softmax(flow_feas / 0.1, dim=2)

        self.flow_xyz = torch.tensor([0,-1,1]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
        flowresults = []
        for i in range(self.steps):
            flow_fea = flow_feas[:, i, :, :, :, :, :]
            flow_step = torch.sum(flow_fea * self.flow_xyz, dim=1)
            if i == 0:
                flowresults.append(flow_step)
            else:
                flowresults.append(warp3d(flowresults[-1], flow_step) + flow_step)

        return flowresults


class Step_model_distill(nn.Module):
    def __init__(self, teacher_model):
        super().__init__()
        self.base_model = teacher_model.base_model
        self.stephead = teacher_model.stephead
        head_dim = teacher_model.steps * teacher_model.field * 3
        self.flowout = nn.Sequential(nn.Conv3d(head_dim, head_dim//2, 3, 1, 1),
                                     nn.LeakyReLU(0.2),
                                     nn.Conv3d(head_dim//2, head_dim//4, 3, 1, 1),
                                     nn.LeakyReLU(0.2),
                                     nn.Conv3d(head_dim//4, head_dim//8, 3, 1, 1),
                                     nn.LeakyReLU(0.2),
                                     nn.Conv3d(head_dim//8, 3, 3, 1, 1))

    def forward(self, x_1, x_2):
        with torch.no_grad():
            encoder_result = self.base_model.encoder(x_1, x_2)
            decoder_result = self.base_model.decoder(encoder_result)

            flow_feas = self.stephead(decoder_result) # x_1.shape[0], self.steps, self.field, 3, *x_1.shape[2:]
        
        flowresult = self.flowout(flow_feas)

        return flowresult


if __name__ == "__main__": 
    base_model = DeepReg().cuda()
    for param in base_model.parameters():
        param.requires_grad = False
    model = Step_model(base_model).cuda()
    x = torch.rand(1, 1, 192, 224, 160).cuda()
    x = torch.tensor(x, requires_grad=True)
    y = model(x, x)
    print(y.requires_grad)
    print(model.base_model.encoder.encoder[0].block[0].weight.requires_grad)
    print(y.shape)

    
    # flow = torch.argmax(flow, dim=1, keepdim=True) # [1,1,3,3,3]
    # y = warp_step(x, flow)