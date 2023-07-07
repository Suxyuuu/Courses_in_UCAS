from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_from_disk, load_dataset
import re


peft_model_id="prompt-tuning-model-20epoch"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)

model = PeftModel.from_pretrained(model, peft_model_id)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


dataset = load_dataset("financial_phrasebank", "sentences_66agree")
print(dataset)
classes = [k.replace("_", " ") for k in dataset["train"].features["label"].names]
dataset = dataset.map(
    lambda x: {"text_label": [classes[label] for label in x["label"]]},
    batched=True,
    num_proc=1,
)

text_column = "sentence"
label_column = "text_label"
max_length = 64

def preprocess_function(examples):
    
    batch_size = len(examples[text_column])
    inputs = [f"\"sentence\" : {examples[text_column][index]}  \"Label\" : " for index in range(batch_size)]
    model_inputs = tokenizer(inputs)

    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        model_inputs["input_ids"][i] = sample_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

    return model_inputs


processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=['sentence', 'label'],
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

print(processed_datasets)


test_dataset = processed_datasets["train"]

device = "cuda"

model.to(device)
model.eval()

print('=====================[Evaluate]==========================')

total_num = 0
hit_num = 0
for item in test_dataset:
    total_num += 1
    # print(item)
    print('[No.{} test case] '.format(total_num), end=' ')
    true_label = item['text_label']

    input_ids = torch.tensor(item['input_ids']).view(-1,len(item['input_ids'])).to(device)
    attention_mask = torch.tensor(item['attention_mask']).view(-1,len(item['attention_mask'])).to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=10, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
        )

        decode_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if decode_outputs[0].find("\"Label\" : positive")!=-1:
            outputs="positive"
        elif decode_outputs[0].find("\"Label\" : negative")!=-1:
            outputs="negative"
        elif decode_outputs[0].find("\"Label\" : neutral")!=-1:
            outputs="neutral"

            
        else:

            print('[Wrong ...]')
            continue
        print('<Predict Label>: {}'.format(outputs), end=' ')
        print('<True Label>: {}'.format(true_label), end=' ')
        if outputs == true_label:
            hit_num += 1
            print('[Right !!!]')
        else:
            print('[Wrong ...]')


print('=====================[Result]============================')
print('Test total num: {}'.format(total_num))
print('Test hit num: {}'.format(hit_num))
print('Test hit accuracy: {}'.format(hit_num/total_num))




