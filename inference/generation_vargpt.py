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

# some instruction examples:
# Please design a drawing of a butterfly on a flower.
# Please create a painting of a black weasel is standing in the grass.
# Can you generate a rendered photo of a rabbit sitting in the grass.
# I need a designed photo of a lighthouse is seen in the distance.
# Please create a rendered drawing of an old photo of an aircraft carrier in the water.
# Please produce a designed photo of a squirrel is standing in the snow.


conversation = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "Please design a drawing of a butterfly on a flower."},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
print(prompt)

inputs = processor(text=prompt, return_tensors='pt').to(0, torch.float32)
model._IMAGE_GEN_PATH = "output.png"
output = model.generate(
    **inputs, 
    max_new_tokens=1000, 
    do_sample=False)

print(processor.decode(output[0], skip_special_tokens=True))
