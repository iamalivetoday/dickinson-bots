# testjesus.py
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

class JesusBot:
    def __init__(self, model_name="./gpt2-jesus"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate_response(self, input_text):
        inputs = self.tokenizer.encode(input_text + self.tokenizer.eos_token, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=150, num_return_sequences=1, pad_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        text = text.split('<STOP>')[0].strip()
        return text

def main():
    bot = JesusBot(model_name="./gpt2-jesus")

    print("JesusBot is ready to chat. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = bot.generate_response(user_input)
        print("JesusBot:", response)

if __name__ == "__main__":
    main()
