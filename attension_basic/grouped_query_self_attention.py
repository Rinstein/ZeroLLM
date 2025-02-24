import torch
from torch import nn


class GQA(nn.Module):
    def __init__(self, input_size, head_num, group_num, embed_size):
        super(GQA, self).__init__()
        self.input_size = input_size
        self.head_num = head_num
        self.group_num = group_num
        self.embed_size = embed_size
        self.head_dim = embed_size // head_num

        assert (self.head_num >= self.group_num) and (self.head_num % self.group_num  == 0),\
          f'head_num need lager/equal and be divisible by group_num, head_num={head_num} group_num={group_num}'  
        assert self.head_dim * head_num == embed_size, 'embed_size must be '\
          f'divisible by head_num, embed_size={embed_size} head_num={head_num}'

        self.k_v_dim = self.head_dim * self.group_num
        self.qkv_proj = nn.Linear(input_size, embed_size + 2 * self.k_v_dim)
        self.out_proj = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.embed_size, self.k_v_dim, self.k_v_dim], dim=-1)

        batch_size, seq_len, input_size  = x.shape
        q = q.reshape(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.group_num, self.head_dim).repeat(1, 1, self.head_num//self.group_num, 1).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.group_num, self.head_dim).repeat(1, 1, self.head_num//self.group_num, 1).transpose(1, 2)

        matrix = (q @ k.transpose(-1, -2)) / torch.tensor(self.head_dim).to(x.device)
        prob = torch.softmax(matrix, dim=-1)
        out = (prob @ v).transpose(1, 2).reshape(batch_size, seq_len, self.embed_size)
        out = self.out_proj(out)

        print('x shape', x.shape)
        print('q shape', q.shape)
        print('k shape', k.shape)
        print('v shape', v.shape)
        print('matrix shape', matrix.shape)
        print('prob shape', prob.shape)
        print('out shape', out.shape)
        return out


if __name__ == "__main__":
    batch_size = 2
    input_size = 12
    head_num = 8
    group_num = 2
    embed_size = 256

    net = GQA(input_size=input_size, head_num=head_num,
              group_num=group_num, embed_size=embed_size)

    seq_len = 20
    input_data = torch.rand(batch_size, seq_len, input_size)
    net(input_data)