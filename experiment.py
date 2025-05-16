import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from huggingface_hub import login
import training_data_collector as other

#model_id = "mistralai/Mistral-7B-v0.1"
#model_id = "EleutherAI/gpt-j-6B"
#model_id = "tiiuae/falcon-7b-instruct"
model_id = "meta-llama/Llama-2-7b-hf"

token = other.token_grabber()
login(token=token)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # or "fp4", "qp4_0" depending on your preference
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",                    # auto-place layers on GPU/CPU
    trust_remote_code=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# Test it out
print(pipe("The quick brown fox", max_new_tokens=32)[0]["generated_text"])
