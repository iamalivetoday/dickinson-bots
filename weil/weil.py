from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    return dataset

def main():
    # Load the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Tokenize and load the dataset
    train_dataset = load_dataset('simone_weil_text.txt', tokenizer)

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Fine-tune the model
    trainer.train()

    # Save the model
    model.save_pretrained('./simone_weil_gpt2')
    tokenizer.save_pretrained('./simone_weil_gpt2')

if __name__ == "__main__":
    main()
