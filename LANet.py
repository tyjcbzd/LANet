import torch.nn as nn
import torch
from backbone.mobileVit import mobile_vit_small
from blocks import DecodingBlock, EFAttention, WeightedBlock
from torchsummary import summary
from thop import profile
class LANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = mobile_vit_small()

        self.att1 = EFAttention(32)
        self.att2 = EFAttention(64)
        self.att3 = EFAttention(96)
        self.att4 = EFAttention(128)
        self.att5 = EFAttention(160)

        self.d_block_1 = DecodingBlock(128, 160, 128)
        self.d_block_2 = DecodingBlock(96, 128, 96)
        self.d_block_3 = DecodingBlock(64, 96, 64)
        self.d_block_4 = DecodingBlock(32, 64, 32)

        self.weight_1 = WeightedBlock(128, 32)
        self.up_1 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.weight_2 = WeightedBlock(96, 32)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.weight_3 = WeightedBlock(64, 32)
        self.up_3 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.weight_4 = WeightedBlock(32, 32)

        self.output_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),  # 1x1
            nn.BatchNorm2d(32),
            # 考虑一下顺序
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 2, kernel_size=1),  # 1x1
            nn.BatchNorm2d(2),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        # torch.Size([16, 32, 128, 128])
        # torch.Size([16, 64, 64, 64])
        # torch.Size([16, 96, 32, 32])
        # torch.Size([16, 128, 16, 16])
        # torch.Size([16, 160, 8, 8])
        # encoder
        enc_1, enc_2, enc_3, enc_4, enc_5 = self.encoder(x)

        # 添加一个注意力模块或者去除冗余模块
        enc_1 = self.att1(enc_1)
        enc_2 = self.att2(enc_2)
        enc_3 = self.att3(enc_3)
        enc_4 = self.att4(enc_4)
        enc_5 = self.att5(enc_5)
        # enc_1 = self.coat1(enc_1)

        # decoder 还需要配置参数
        # torch.Size([16, 128, 16, 16])
        # torch.Size([16, 96, 32, 32])
        # torch.Size([16, 64, 64, 64])
        # torch.Size([16, 32, 128, 128])
        dec_1 = self.d_block_1(enc_4, enc_5)
        dec_2 = self.d_block_2(enc_3, dec_1)
        dec_3 = self.d_block_3(enc_2, dec_2)
        dec_4 = self.d_block_4(enc_1, dec_3)

        weight_1 = self.up_1(self.weight_1(dec_1))
        weight_2 = self.up_2(weight_1 + self.weight_2(dec_2))
        weight_3 = self.up_3(weight_2 + self.weight_3(dec_3))
        weight_4 = weight_3 + self.weight_4(dec_4)

        # 最终预测
        pred = self.output_conv(weight_4)
        # return edge_pred, pred
        return pred

    def load_encoder_weight(self):
        # One could get the pretrained weights via PyTorch official.
        self.encoder.load_state_dict(torch.load("backbone/mobilevit_s.pt", map_location=torch.device('cuda')))


if __name__ == '__main__':
    # print(torch.has_mps)

    # mbs = res2net50_v1b_26w_4s(pretrained=False)

    device = torch.device('cuda')
    images = torch.rand(16, 3, 256, 256).to(device)
    model = LANet().to(device)
    model.load_encoder_weight()

    pred = model(images)

    print(pred.shape)

    input_shape = (3, 256, 256)  # 替换为你的输入张量形状
    summary(model, input_shape)
    input_data = torch.randn((16,3,256,256)).to(device)

    flops, params = profile(model, inputs=(input_data,))
    print(f"FLOPs: {flops/1e9}")

    # mbs = LeeSeg()
    # # check_point = torch.load("mobilevit_s.pt", map_location=torch.device('cpu'))
    # # mbs.load_state_dict(check_point)
    # # 定义输入的样本尺寸
    # input_size = (1, 3, 256, 256)
    #
    # # 使用thop库计算模型FLOPs
    # input_data = torch.randn(input_size)
    # flops, params = profile(mbs, inputs=(input_data,))
    #
    # print(f"FLOPs: {flops} ")
    # print(f"Params: {params} ")
