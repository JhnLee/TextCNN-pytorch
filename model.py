import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseTextCNN(nn.Module):

    def __init__(self, num_classes,
                 vocab_size,
                 embedding_size,
                 pre_weight=None,
                 pad_index=0):

        super(BaseTextCNN, self).__init__()
        self.layers = nn.Sequential(
            MultiChannelEmbedding(vocab_size,
                                  embedding_size,
                                  pre_weight,
                                  pad_index),
            Convolution(in_channels=150,
                        out_channels=300),
            MaxOverTimePooling(),
            nn.Dropout(),
            nn.Linear(300, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


class MultiChannelEmbedding(nn.Module):
    def __init__(self, vocab_size,
                 embedding_size,
                 pre_weight=None,
                 pad_index=0):

        super(MultiChannelEmbedding, self).__init__()
        self.pre = pre_weight

        if self.pre is not None:
            self.static = nn.Embedding.from_pretrained(torch.from_numpy(self.pre),
                                                       freeze=True)
            self.non_static = nn.Embedding.from_pretrained(torch.from_numpy(self.pre),
                                                           freeze=False)
        else:
            self.static = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_index)

    def forward(self, x):
        static = self.static(x).permute(0, 2, 1).float()  # (B, L, H) -> (B, H, L)
        if self.pre is not None:
            non_static = self.non_static(x).permute(0, 2, 1).float()
            return static, non_static
        return static  # (B, H, L)


class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Convolution, self).__init__()
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=in_channels,
                                             out_channels=out_channels // 3,
                                             kernel_size=n) for n in range(3, 6)])

    def forward(self, x):
        # For pretrained
        if type(x) is tuple:
            feature_maps = [F.relu(c(x[0])) + F.relu(c(x[1])) for c in self.conv]  # static & non static
        else:
            feature_maps = [F.relu(c(x)) for c in self.conv]  # static
        return feature_maps  # (B, H, L) -> (B, C, L) x 3    (H: in_channel, C: out_channel)


class MaxOverTimePooling(nn.Module):
    def forward(self, x):
        fmaps = [f.max(dim=-1)[0] for f in x]  # (B, C) x 3
        return torch.cat(fmaps, dim=-1)  # (B, 3C)


