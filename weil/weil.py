from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import os

def load_dataset(file_path, tokenizer, block_size=128):
    """
    Load and tokenize the dataset using the provided tokenizer.
    
    Args:
        file_path (str): Path to the text file containing the dataset.
        tokenizer: Pre-trained GPT-2 tokenizer.
        block_size (int): Size of each block of tokens.

    Returns:
        TextDataset: A dataset ready for language model training.
    """
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    return dataset

def concatenate_text_files(directory, output_file):
    """
    Concatenate all text files in the given directory into a single text file.
    
    Args:
        directory (str): Directory containing text files.
        output_file (str): Output file path where the concatenated text will be saved.
    """
    with open(output_file, 'w') as outfile:
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                with open(os.path.join(directory, filename), 'r') as infile:
                    outfile.write(infile.read() + '\n')

def main():
    # Define the directory containing text files and the output concatenated file
    #text_directory = './text'
    #concatenated_text_file = 'simone_weil_combined_text.txt'

    # Step 1: Concatenate all text files into one
    #concatenate_text_files(text_directory, concatenated_text_file)

    # Step 2: Load the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Step 3: Tokenize and load the dataset
    train_dataset = load_dataset('simone_weil_combined_text.txt', tokenizer)

    # Step 4: Data collator for language modeling (no masking)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
    )

    # Step 5: Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=3,  # Start with 3 epochs and adjust if necessary
        per_device_train_batch_size=1,  # Adjust based on your GPU/CPU capacity
        save_steps=10_000,
        gradient_accumulation_steps=4,  # Accumulates gradients over 4 steps to simulate batch size of 4
        save_total_limit=2,
        learning_rate=0.00001,  # Lower learning rate for careful fine-tuning
        weight_decay=0.01,  # Helps to avoid overfitting
        prediction_loss_only=True,
    )


    # Step 6: Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Step 7: Fine-tune the model
    trainer.train()

    # Step 8: Save the fine-tuned model and tokenizer
    model.save_pretrained('./simone_weil_gpt2')
    tokenizer.save_pretrained('./simone_weil_gpt2')

if __name__ == "__main__":
    main()
