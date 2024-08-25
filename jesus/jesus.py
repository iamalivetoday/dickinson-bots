# jesus.py
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset
import torch

class JesusBot:
    def __init__(self, model_name="gpt2", data_file="jesus_cleaned.txt"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({'pad_token': '<|pad|>', 'additional_special_tokens': ['<STOP>']})
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.data_file = data_file

    def load_and_tokenize_data(self):
        data_files = {"train": self.data_file}
        dataset = load_dataset("text", data_files=data_files)

        def tokenize_function(examples):
            tokenized_output = self.tokenizer(examples["text"], padding="max_length", truncation=True)
            tokenized_output["labels"] = tokenized_output["input_ids"].copy()
            return tokenized_output

        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        return tokenized_datasets

    def fine_tune(self, tokenized_datasets):
        training_args = TrainingArguments(
            output_dir="./gpt2-jesus",
            evaluation_strategy="no",  # Disable evaluation
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            num_train_epochs=3,
            weight_decay=0.01,
            save_total_limit=1,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
        )

        trainer.train()

        # Save the model and tokenizer
        self.model.save_pretrained("./gpt2-jesus")
        self.tokenizer.save_pretrained("./gpt2-jesus")

    def generate_response(self, input_text):
        inputs = self.tokenizer.encode(input_text + self.tokenizer.eos_token, return_tensors='pt')
        # Generate response with attention mask to handle padding properly
        outputs = self.model.generate(
            inputs,
            max_length=150,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,  # Handle padding correctly
            attention_mask=inputs.ne(self.tokenizer.pad_token_id)  # Correct attention mask
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        text = text.split('<STOP>')[0].strip()
        return text

# Function to create and fine-tune the JesusBot
def create_and_train_bot(data_file="jesus_cleaned.txt"):
    bot = JesusBot(data_file=data_file)
    tokenized_datasets = bot.load_and_tokenize_data()
    bot.fine_tune(tokenized_datasets)
    return bot

if __name__ == "__main__":
    bot = create_and_train_bot()
