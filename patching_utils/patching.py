
from patching_utils.vargpt_sample import vargpt_sample
def patching (model):
    setattr(model, "_sample", vargpt_sample.__get__(model))
