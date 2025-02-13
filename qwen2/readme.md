## Qwen模型解析
### 1. 前置知识
transformers repo：https://github.com/huggingface/transformers<br>
可以简单看看huggingface开源的transformers库，要是不看，直接看代码也行。<br>
该库可以让我们很方便的 chat with 各种开源大模型。其文档可以帮助我们快速了解和熟悉LLM的世界。<br>
### 2. 先把代码跑起来
这里参考[qwen2.5](https://github.com/QwenLM/Qwen2.5)的Quick Start，并修改一下为小一点的模型。<br>
环境搭建装一个transformers和torch就行，qwen2.5建议装新一点的python&transformers&torch, 我这边的版本如下：
```
Python 3.13.2
transformers 4.48.3
torch 2.6.0
```
注：版本应该大差不差就行，能跑起来就应该没啥问题。
直接跑代码, 然后可以看到模型的输入与输出了：
```
python qwen2/hello_qwen2.py
```
注：transformers会下载预训练的模型到本地，位置为**~/.cache/huggingface/hub**

### 3. 先理解一下LLM的大致过程
LLM的结构其实不复杂，大致分为如下几步：
1. Tokenize：
将文字转化为词表中对应的下标索引，模型只能输入数字的信息，在模型中这个索引就是表示对应的文字信息。
2. Predict next token：
将转化好的token列表输入模型，预测得到下一个token！这个就是qwen2模型做的事情，大部分LLM也是这样的。
3. Predict next next token：
将上一步中预测的token放入原来的token列表末尾，再次将token列表放入模型中，再得到一个新token！
4. Repeat predict next ... token：
模型一直预测下一个token，直到模型输出停止或者达到最大输出长度。
5. De-tokenize：
将新输出的token列表转化为词表中原始的含义，拼接起来，就得到了LLM给你的回答了！

### 4. 分析模型结构
可参考DataWhale的开源教程[Qwen Blog](https://github.com/datawhalechina/tiny-universe/blob/main/content/Qwen-blog/readme.md)。<br>
首先来分析一下模型如何预测下一个token，通过运行脚本**prepare_for_predict_one_token.py**，并以debug模式进入到Qwen2Model.forward()函数，就可以逐步分析qwen2的模型结构了：
<center><img src=assets/Qwen2Model_forward.png alt=Qwen2Model_forward width=40%></center>
如果在VS code中无法进入Qwen2Model，需要配制一个debug setting，加入参数**"justMyCode": false**：
<center><img src=assets/VS_Code_debug_setting.png alt=VS_Code_debug_setting width=40%></center>
qwen2的模型结构如下所示(from DataWhale Qwen Blog)，Qwen2Model仅包含图中虚线框部分，输入的预处理和最后接的linear head不在该模型内：
<center><img src=assets/Qwen2Model_structure.png alt=Qwen2Model_structure width=80%></center>
Qwen2Model模型接收的是token向量，表示输入的提示文本，输出的是一个编码后的特征，在下游任务中，使用该特征接一些线性层和softmax即可得到预测下一个token的概率。
以**prepare_for_predict_one_token.py**的输入为例，输入shape为(1, 20), 输出的last_hidden_state shape为(1, 20, 2048)，这里的2048即为config中设置的hidden_size。

### 5. 如何预测**下下**个token？
上一节中，得到第一个预测的token看起来不难，不过如何得到预测的第二个token呢？理解这个过程，会帮助我们理解LLM的decoder-only结构，KV cache等过程。<br>
一个大概的过程是，得到第一个预测的token后，将这个token拼接在之前的token列表最后，然后一起送入Qwen2Model中再得到第二个预测的token。看起来是这样子，但是细节有一些问题：
1. 难道我们需要完全重跑Qwen2Model？如果预测每个token都完全跑一遍模型，似乎时延有点大
2. 直觉上，除了我们最后新加的这个token，前面其余部分是否是重复计算？
3. 
todo：分析预测第二个token使用的kV cache机制，怎么做的，为什么可以这么做？分析generate的机制

### 6. 一些细节上的探究
#### 6.1 RoPE
#### 6.2 Qwen2RMSNorm