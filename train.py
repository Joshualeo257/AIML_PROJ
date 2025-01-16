from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd

dataset = load_dataset("csv", data_files=r"simplified_emotions.csv")

dataset = dataset['train'].train_test_split(test_size=0.2)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained("emotion_model", num_labels=28)

emotion_labels = list(set(dataset['train']['Emotion'])) 
label2id = {emotion: idx for idx, emotion in enumerate(emotion_labels)}  
id2label = {idx: emotion for emotion, idx in label2id.items()}  


model.config.label2id = label2id
model.config.id2label = id2label


def preprocess_function(examples):
    tokenized_inputs = tokenizer(
        examples["Review"],  
        padding="max_length",
        truncation=True,
        max_length=150,
        return_tensors="pt"
    )
    
    tokenized_inputs['labels'] = [label2id[emotion] for emotion in examples['Emotion']]
    return tokenized_inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True)


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)


trainer.train()


model.save_pretrained("emotion_model")
tokenizer.save_pretrained("emotion_model")
