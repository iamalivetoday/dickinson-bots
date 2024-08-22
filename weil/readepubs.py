import os
from ebooklib import epub  # Ensure this import is present
from ebooklib import ITEM_DOCUMENT  # Ensure ITEM_DOCUMENT is imported
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import nltk

# Download the punkt tokenizer model
nltk.download('punkt')
nltk.download('punkt_tab')

def read_epub(file_path):
    book = epub.read_epub(file_path)  # Ensure epub is correctly referenced
    text = []
    
    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:  # Use ITEM_DOCUMENT correctly
            soup = BeautifulSoup(item.get_body_content(), 'html.parser')
            text.append(soup.get_text())
    
    return '\n'.join(text)

def process_epubs_in_folder(folder_path):
    all_tokens = []
    epub_files = [f for f in os.listdir(folder_path) if f.endswith('.epub')]

    # Using tqdm to create a progress bar
    for filename in tqdm(epub_files, desc="Processing EPUB files"):
        print(f"\nEntering file: {filename}")  # Stub for entering file
        file_path = os.path.join(folder_path, filename)
        
        epub_text = read_epub(file_path)
        
        # Tokenize the extracted text
        tokens = word_tokenize(epub_text)
        
        # Append the tokens to the list
        all_tokens.extend(tokens)
        
        print(f"Leaving file: {filename}")  # Stub for leaving file
    
    # Save the tokens to a file
    with open("simone_weil_text.txt", "w") as f:
        f.write(' '.join(all_tokens))

    print("Tokenized text saved to simone_weil_text.txt")

# Replace 'weilbooks' with the path to your folder
folder_path = 'weilbooks'
process_epubs_in_folder(folder_path)
