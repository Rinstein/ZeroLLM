from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-0.5B-Instruct" #"Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "现在国内有哪些比较强的大模型厂商？"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(res)
'''
某次回答如下所示：
当前，国内有多个大模型厂商。其中，阿里云和百度公司是中国的两大巨头，它们分别拥有大量的超大规模语言模型和深度学习技术。此外，腾讯、阿里巴巴、京东等公司也在积极研发和发展自己的大模型技术和产品。这些厂商不仅在人工智能领域取得了显著的成绩，也对社会产生了深远的影响。
'''