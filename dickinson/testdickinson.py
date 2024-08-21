from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('./emily_dickinson_gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, max_length=100, temperature=0.75, top_k=400, top_p=0.85, repetition_penalty=1.3):
    inputs = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True, padding=True)
    output = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        do_sample=True,  # Enable sampling for more variety
        temperature=temperature,  # Control randomness
        top_k=top_k,  # Limit to top K tokens
        top_p=top_p,  # Nucleus sampling to control the diversity
        repetition_penalty=repetition_penalty,  # Avoid repetition
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Test with some example prompts
print("Prompt: What is hope?")
print("Response:", generate_response("What is hope?"))

print("Prompt: What's after death?")
print("Response:", generate_response("What's after death?"))

print("Prompt: Should slavery be abolished?")
print("Response:", generate_response("Should slavery be abolished?"))

print("Prompt: Should I download Tinder?")
print("Response:", generate_response("Should I download Tinder?"))
