import pandas as pd 
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate
import numpy as np
from transformers import TrainingArguments, Trainer

df = pd.read_csv("/mnt/data/compexp/nli/data/analysis/snli_1.0_dev.txt", sep = '\t')

# filter out rows where people reached no consensus on the annotation by getting rid of all rows with label "-"
df = df[df['gold_label'] != "-"].copy()

#define mappign (entailment, neutral, contradiction)
label_map = {"entailment" : 0, "neutral": 1, "contradiction": 2}
#create a new column for the label and populate using mapping from above
df['label'] = df['gold_label'].str.lower().map(label_map)

#create a new pandas df that takes sentence1, sentence2, and the label 
dataset = Dataset.from_pandas(df[['sentence1', 'sentence2', 'label']])



# tokenize the dataset

#load
model_path = "/mnt/data/gemma-2-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_path)

#tokenizer.pad_token = tokenizer.eos_token #needed because gemma no like empty spaces. Adds padding

def preprocess_nli(examples): 
    return tokenizer(

        examples["sentence1"],
        examples["sentence2"], 
        #truncation = True, 
        max_length = 70,
        #padding = "max_length" #adds the padding 

    )

# batched = true means there will be a kind of placeholder token inserted so we have that perfect rectangle arr shape for the GPU
tokenized_data = dataset.map(preprocess_nli, batched = True)

#batch organizer (GPU cant handle "jagged" data, it needs a perfect rectangle of number)

#data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#bnb config to allow training the model on a weaker GPU by compressing the weights and only un-compressing in small batches so the GPU vram doesnt get overwhelmed
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtypes = torch.bfloat16,
)


# load model

model = AutoModelForSequenceClassification.from_pretrained(

model_path,
num_labels = 3,
quantization_config=bnb_config, #apply the compresssion for bnb
device_map = "auto",

)

# Add LoRA adapters for classification
#model.config.pad_token_id = tokenizer.pad_token_id

peft_config = LoraConfig(

task_type ="SEQ_CLS",
r = 16, 
lora_alpha = 32,
target_modules = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "down_proj", "up_proj"],
)

model = prepare_model_for_kbit_training(model)

model = get_peft_model(model, peft_config)

#accuracy

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis = -1)
    return accuracy.compute(predictions=predictions, references = labels)



training_args = TrainingArguments(
    output_dir = "/mnt/data/train_test",
    #per_device_train_batch_size = 4,
    #gradient_accumulation_steps = 4, 
    #learning_rate = 2e-5,
    #num_train_epochs = 1,
    #logging_steps = 10,
    #save_strategy = "epoch",
    #fp16 = True, 
)



trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_data,
    processing_class = tokenizer,
    compute_metrics = compute_metrics,
)


trainer.train()

model.save_pretrained("/mnt/data/gemma-tuned")
tokenizer.save_pretrained("/mnt/data/gemma-tuned")

# Now try to evaluate
metrics = trainer.evaluate(eval_dataset=tokenized_data)

print(f"Final Accuracy: {metrics['eval_accuracy'] * 100:.2f}%")
