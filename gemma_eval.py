from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm 



df = pd.read_csv("compexp/nli/data/analysis/snli_1.0_dev.txt", sep='\t') #the sep because everything is tab seperated in dev.txt


gemini_predic_label = []

total_count = 0

correct_count = 0



tokenizer = AutoTokenizer.from_pretrained("/mnt/data/gemma-2-2b-it")#/mnt/data/gemma-7b-it
model = AutoModelForCausalLM.from_pretrained("/mnt/data/gemma-2-2b-it", device_map="auto")



for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Progress"):


    actual = str(row['gold_label']).lower().strip()

    if actual == "-":
        continue


#for index, row in df.iterrows():

    premise = row['sentence1']
    hypothesis = row['sentence2']

    input_text = f"""<start_of_turn>user\n
Classify the relationship between these two sentences as exactly one of: entailment, neutral, or contradiction. 

Premise: {premise}
Hypothesis: {hypothesis}

Output only the label.<end_of_turn>\n
<start_of_turn>model\n
"""
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids, max_new_tokens=10)
    

    #splicing string so we get only the label

    #we want everyting after this 

    target = "<start_of_turn>model\n"

    raw = tokenizer.decode(outputs[0])#skip_special_tokes = true

    index = raw.find(target)

    prediction = raw[index + len(target):].strip()

    gemini_predic_label.append(prediction)

#print(gemini_predic_label[0])


    clean_prediction = prediction.lower().strip()

    total_count+=1
    if actual in clean_prediction:
        correct_count += 1


if total_count > 0:
    accuracy = (correct_count/total_count) * 100
    print(f"Final accuracy: {accuracy:.2f}%")

else: 
    print("err")