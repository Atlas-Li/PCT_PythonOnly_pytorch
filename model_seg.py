import torch
import torch.nn as nn
import torch.nn.functional as F


class PCT_seg(nn.Module):
    def __init__(self, part_num=50):
        super(PCT_seg, self).__init__()
        self.part_num = part_num
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(128)

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )

        self.label_conv = nn.Sequential(
            nn.Conv1d(self.part_num, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )

        self.convs1 = nn.Conv1d(1024 * 3 +64, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.part_num, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()


    def forward(self, x, cls_label):
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))  # B, D, N
        x1 = self.sa1(x)
        x2 = self.sa1(x1)
        x3 = self.sa1(x2)
        x4 = self.sa1(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)
        x_max = torch.max(x, 2)
        x_avg = torch.mean(x, 2)

        x_max_feature = x_max[0].view(batch_size, -1).unsqueeze(-1).repeat(1, 1,N)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1,N)
        cls_label_one_hot = cls_label.view(batch_size, self.part_num, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        
        x_global_feature = torch.cat((x_max_feature, x_avg_feature, cls_label_feature), 1)
        x = torch.cat((x, x_global_feature), 1)
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        
        return x



class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels//4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels//4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)   # b, n, c
        x_k = self.k_conv(x)                    # b, c, n
        x_v = self.v_conv(x)                    # b, c, n

        energy = torch.matmul(x_q, x_k)   # b, n, n
        attn = self.softmax(energy)
        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))

        x_r = torch.matmul(x_v, attn)   # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x-x_r)))
        x = x + x_r

        return x





if __name__ == "__main__":
    sa = PCT_seg(100)
    input = torch.randn(2, 3, 100)
    # sa(input)

    input = torch.randn(2, 3, 100)
    label_input = torch.randn(2, 1, 100)
    sa(input, label_input)
