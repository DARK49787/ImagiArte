import gradio as gr
import ollama
from PIL import Image
import os
import torch
import cv2
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler, ControlNetModel
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class ImageEnhancer:
    def __init__(self):
       
        self.temp_path = r"C:\Users\pichau\Documents\Projeto-drack-ia\temp"
        os.makedirs(self.temp_path, exist_ok=True)

     
        self.model_path = r"C:\Users\pichau\Documents\Projeto-drack-ia\ComfyUI\models\checkpoints\dreamshaper_8.safetensors"

        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo não encontrado em: {self.model_path}")

        
        torch.cuda.empty_cache()

       
        self.blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        )
      
        if torch.cuda.is_available():
            self.blip_model = self.blip_model.to("cuda")

      
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_scribble",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

      
        try:
            self.pipe = StableDiffusionImg2ImgPipeline.from_single_file(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                controlnet=self.controlnet,
            )
        except Exception as e:
            print(f"Erro ao carregar o modelo local: {str(e)}")
            print("Usando modelo padrão do Hugging Face: runwayml/stable-diffusion-v1-5")
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                use_safetensors=True,
                controlnet=self.controlnet,
            )

        
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()  

    def generate_scribble_map(self, image):
        
        img_array = np.array(image.convert("RGB"))

        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        
        edges = cv2.Canny(gray, 100, 200)

      
        scribble = 255 - edges


        scribble_image = Image.fromarray(scribble)
        return scribble_image

    def describe_image(self, image):
        try:
            inputs = self.blip_processor(images=image, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            outputs = self.blip_model.generate(**inputs, max_length=200)
            description = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
            return description
        except Exception as e:
            return f"Erro ao descrever a imagem: {str(e)}"

    def refine_prompt(self, description, image):
        temp_image_path = os.path.join(self.temp_path, "temp_refine_image.jpg")
        image.save(temp_image_path)

        try:
            refined_prompt = ollama.generate(
                model="llava",
                prompt=f"Refine this image description into a detailed prompt, enhancing the visual details without adding new elements, keeping it suitable for a anime: {description}",
                images=[temp_image_path]
            )["response"]

            additional_instructions = "anime style, high detail,realistic, enhanced clarity, improved lighting"
            max_tokens = 700
            refined_prompt_words = refined_prompt.split()
            additional_words = additional_instructions.split()
            total_words = len(refined_prompt_words) + len(additional_words)

    
            if total_words > max_tokens:
                allowed_words = max_tokens - len(additional_words)
                refined_prompt = " ".join(refined_prompt_words[:allowed_words])
            
            refined_prompt += additional_instructions
            return refined_prompt
        except Exception as e:
            return f"Erro ao refinar o prompt: {str(e)}"
        finally:
            
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

    def enhance_image(self, image, description):
        if "Erro ao descrever a imagem" in description:
            return None, description

        
        prompt = self.refine_prompt(description, image)
        if "Erro ao refinar o prompt" in prompt:
            return None, prompt

        try:
           
            image = image.resize((512, 512))  

          
            scribble_map = self.generate_scribble_map(image)

            
            image = self.pipe(
                prompt=prompt,
                negative_prompt="texture to the sketch,draw,artwork,distorted, blurry, low quality, extra objects, unrealistic elements",
                image=image,
                controlnet_conditioning_image=scribble_map,
                controlnet_conditioning_scale=1,  
                num_inference_steps=9,  
                strength=0.7, 
            ).images[0]

           
            torch.cuda.empty_cache()

            return image, f"Imagem aprimorada com sucesso! Prompt: {prompt}"
        except Exception as e:
            return None, f"Erro ao aprimorar a imagem: {str(e)}"

    def process_image(self, input_image):
        if input_image is None:
            return None, "Por favor, carregue uma imagem."

        
        description = self.describe_image(input_image)
        if "Erro" in description:
            return None, description

     
        enhanced_image, message = self.enhance_image(input_image, description)
        return enhanced_image, message


app = ImageEnhancer()


with gr.Blocks() as demo:
    gr.Markdown("#ImagiArte: Quando a IA Dá Forma ao Seu Desenho  (Conexão TechFest)")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Carregar Imagem de Entrada")
            enhance_button = gr.Button("Aprimorar Imagem")
        
        with gr.Column():
            output_image = gr.Image(label="Imagem Aprimorada")
            output_message = gr.Textbox(label="Mensagem")

    enhance_button.click(
        fn=app.process_image,
        inputs=[input_image],
        outputs=[output_image, output_message]
    )


demo.launch(share=False )