import torch
from torch import nn

"""
Cross attention shape need this condition：
query: [batch_size, query_seq_len, query_dim]
key: [batch_size, value_seq_len, query_dim]
value: [batch_size, value_seq_len, value_dim]
condition: query_dim == key_dim and key_seq_len == value_seq_len
so, query seq len can be choose by your self 
"""
class CMHA(nn.Module):
    def __init__(self, input_size, head_num, embed_size):
        super(CMHA, self).__init__()
        self.input_size = input_size
        self.head_num = head_num
        self.embed_size = embed_size
        self.head_dim = embed_size // head_num

        assert self.head_dim * head_num == embed_size, 'embed_size must be '\
          f'divisible by head_num, embed_size={embed_size} head_num={head_num}'

        self.query_proj = nn.Linear(input_size, embed_size)
        self.key_proj = nn.Linear(input_size, embed_size)
        self.value_proj = nn.Linear(input_size, embed_size)
        self.out_proj = nn.Linear(embed_size, embed_size)
    
    def forward(self, q, k, v, mask=None):
        print('q before projection shape', q.shape)
        print('k before projection shape', k.shape)
        print('v before projection shape', v.shape)
        q = self.query_proj(q)
        k = self.key_proj(k)
        v = self.value_proj(v)

        batch_size, _, _  = q.shape
        q = q.reshape(batch_size, -1, self.head_num, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, -1, self.head_num, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, -1, self.head_num, self.head_dim).transpose(1, 2)

        matrix = (q @ k.transpose(-1, -2)) / torch.tensor(self.head_dim).to(q.device)
        if mask is not None:
            matrix = matrix + mask[:, None, None, :] * -1e9
        prob = torch.softmax(matrix, dim=-1)
        out = (prob @ v).transpose(1, 2).reshape(batch_size, -1, self.embed_size)
        out = self.out_proj(out)

        print('q after projection shape', q.shape)
        print('k after projection shape', k.shape)
        print('v after projection shape', v.shape)
        print('matrix shape', matrix.shape)
        print('prob shape', prob.shape)
        print('out shape', out.shape)
        return out


if __name__ == "__main__":
    batch_size = 2
    input_size = 12
    head_num = 8
    embed_size = 256

    net = CMHA(input_size=input_size, head_num=head_num, embed_size=embed_size)

    seq_len = 20
    mask = torch.zeros(batch_size, seq_len)
    mask[:, -1:] = 1
    q = torch.rand(batch_size, 10, input_size)
    k = torch.rand(batch_size, seq_len, input_size)
    v = torch.rand(batch_size, seq_len, input_size)
    net(q, k, v, mask)

    # more, model query and key embed_dim can be change to same value