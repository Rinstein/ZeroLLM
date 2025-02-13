import torch
from transformers.models.qwen2 import Qwen2Config, Qwen2Model


def run_qwen2():
    config = Qwen2Config(
        vocab_size=151936,
        hidden_size=4096 // 2,
        intermediate_size=22016 // 2,
        num_hidden_layers=32 // 2,
        num_attention_heads=32,
        max_position_embeddings=2048 // 2
    )
    model = Qwen2Model(config)

    input_ids = torch.randint(0, config.vocab_size, (1, 20))

    # here only get encoding result for input_ids
    res = model(input_ids)
    print(res.last_hidden_state.shape)
    '''
    print result torch.Size([1, 20, 2048])
    correspond to input_ids.shape = (1, 20)
    '''


if __name__ == "__main__":
    run_qwen2()
