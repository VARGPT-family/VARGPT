import requests
from PIL import Image

import torch
from transformers import AutoProcessor, AutoTokenizer
from vargpt_llava.modeling_vargpt_llava import VARGPTLlavaForConditionalGeneration
from vargpt_llava.prepare_vargpt_llava import prepare_vargpt_llava 
from vargpt_llava.processing_vargpt_llava import VARGPTLlavaProcessor
from patching_utils.patching import patching

model_id = "VARGPT-family/VARGPT_LLaVA-v1"
prepare_vargpt_llava(model_id)

model = VARGPTLlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float32, 
    low_cpu_mem_usage=True, 
).to(0)
patching(model)

tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = VARGPTLlavaProcessor.from_pretrained(model_id)

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "Please explain the meme in detail."},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
image_file = "./assets/llava_bench_demo.png"
print(prompt)

raw_image = Image.open(image_file)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float32)

output = model.generate(
    **inputs, 
    max_new_tokens=2048, 
    do_sample=False)

print(processor.decode(output[0], skip_special_tokens=True))
