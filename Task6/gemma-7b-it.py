from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/mnt/data/gemma-7b-it")
model = AutoModelForCausalLM.from_pretrained("/mnt/data/gemma-7b-it", device_map="auto")

input_text = "True or False: An IPO is when a private company first offers stock to the public."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_length=2000)
print(tokenizer.decode(outputs[0]))
