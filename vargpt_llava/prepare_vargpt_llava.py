from .configuration_vargpt_llava import VARGPTLlavaConfig
from .processing_vargpt_llava import VARGPTLlavaProcessor
from .modeling_vargpt_llava import VARGPTLlavaForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, CLIPVisionConfig, CLIPVisionModel, AutoTokenizer, AutoImageProcessor, CLIPImageProcessor, AutoConfig
from transformers import AutoProcessor, LlavaProcessor, GenerationConfig, LlavaForConditionalGeneration
import torch
from transformers import AutoProcessor
from transformers.processing_utils import ProcessorMixin


vargpt_save_path = "VARGPT-family/VARGPT_LLaVA-v1" 


def check_file_exists(directory, filename):
    import os
    file_path = os.path.join(directory, filename)
    return os.path.isfile(file_path)


def prepare_vargpt_llava(save_path=vargpt_save_path, prepared_modules=["model", "tokenizer", "processor", "image_processor"], device=None):


    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    


    vargpt_llava_config = VARGPTLlavaConfig.from_pretrained(save_path)

    processor = VARGPTLlavaProcessor.from_pretrained(vargpt_save_path)
    # register into hugginface
    AutoConfig.register(vargpt_llava_config.model_type, VARGPTLlavaConfig)
    AutoModelForVision2Seq.register(VARGPTLlavaConfig, VARGPTLlavaForConditionalGeneration)
    AutoProcessor.register(VARGPTLlavaConfig, processor_class=VARGPTLlavaProcessor)

   
    
if __name__ == "__main__":
    prepare_vargpt_llava()