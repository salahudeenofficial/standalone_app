import torch
import numpy as np


class Pipeline:
    def __init__(self,models_dir="models"):
        self.models_dir = models_dir
    def run_pipeline(self,unet_path,clip_path,vae_path,lora_path,positive_prompt="",negative_prompt=""
                    control_video_path=None,reference_image_path=None,width=480,height=480,length=37,batch_size=1,strength=1.0,
                    seed=270400132721985,
                    steps=4,
                    cfg=1.0,
                    sampler_name="ddim",
                    scheduler="normal",
                    denoise=1.0,
                    output_path="output.
                    mp4"
                    ):
        #loading all the models,Control_video and reference_image

        try:
            #testing va_de encoding 
            vae_state_dict = utils.load_torch_file(vae_path)
            vae  = 
            