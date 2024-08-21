from transformers import GPT2Tokenizer
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the padding token to the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token

# Load the text files
with open('letters.txt', 'r', encoding='utf-8') as letters_file:
    letters_text = letters_file.read()

with open('poems.txt', 'r', encoding='utf-8') as poems_file:
    poems_text = poems_file.read()

# Combine the texts
combined_text = letters_text + "\n\n" + poems_text

inputs = tokenizer(combined_text, return_tensors='pt', max_length=512, truncation=True, padding=True)

class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx]}

# Create the dataset
dataset = CustomDataset(inputs['input_ids'], inputs['attention_mask'])

# Load the GPT-2 model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=5,  # Start with 3 epochs and adjust if necessary
    per_device_train_batch_size=2,  # Adjust based on your GPU/CPU capacity
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=0.001,  # Lower learning rate for careful fine-tuning
    weight_decay=0.01,  # Helps to avoid overfitting
    prediction_loss_only=True,
)

# Data collator to handle padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Start training
trainer.train()

# Save the model
trainer.save_model('./emily_dickinson_gpt2')
