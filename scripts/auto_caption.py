import os
import gc
import traceback
from pathlib import Path

def auto_caption_dataset(dataset_dir: str, category: str = "style"):
    print(f"--- Starting LLaVA v1.5 7B Auto-Captioning for {dataset_dir} (Category: {category}) ---", flush=True)

    try:
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        from PIL import Image
        import torch
    except ImportError as e:
        print(f"Warning: Missing dependencies for auto-captioning. Skipping. Error: {e}", flush=True)
        return

    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images_to_process = []
    total_images_found = 0

    # Find all images to enthusiastically replace any existing .txt files with LLaVA
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            file_path = Path(os.path.join(root, file))
            if file_path.suffix.lower() in image_extensions:
                total_images_found += 1
                images_to_process.append(file_path)

    print(f"Scanned directory recursively. Total image files found to caption: {total_images_found}", flush=True)

    if not images_to_process:
        print("No images found in the dataset directory! Skipping LLaVA.", flush=True)
        return

    print(f"Loading LLaVA v1.5 7B...", flush=True)
    try:
        local_dir = "/opt/models/llava"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        print(f"Loading local offline LLaVA model from {local_dir}...", flush=True)

        processor = AutoProcessor.from_pretrained(local_dir, use_fast=False)
        model = LlavaForConditionalGeneration.from_pretrained(
            local_dir, 
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(device)
        model.eval()

        for img_path in images_to_process:
            print(f"Captioning {img_path.name}...", flush=True)
            try:
                # Read original caption
                txt_path = img_path.with_suffix('.txt')
                original_caption = ""
                if txt_path.exists():
                    with open(txt_path, "r", encoding="utf-8") as f:
                        original_caption = f.read().strip()

                if category.lower() == "person":
                    instruction = f"The original caption is: '{original_caption}'. Analyze this image of a person. Generate an improved, highly detailed caption based on the visual evidence. You MUST specifically describe their emotion (e.g., happy, sad, smiling), their clothing, and the background setting (e.g., beach, room, city). Incorporate the original caption organically."
                else:
                    instruction = f"The original caption is: '{original_caption}'. Analyze this artistic image. Generate an improved, highly detailed caption. You MUST clearly communicate the distinctive visual characteristics of this style. Focus on specific visual elements, textures, colors, and artistic techniques used. Incorporate the original caption organically."

                base_prompt = f"USER: <image>\n{instruction}\nASSISTANT:"

                raw_image = Image.open(img_path).convert("RGB")
                inputs = processor(text=base_prompt, images=raw_image, return_tensors="pt").to(device, dtype)

                with torch.no_grad():
                    generate_ids = model.generate(**inputs, max_new_tokens=300)
                
                # Decode output, removing the prompt from the final string
                output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                caption = output.split("ASSISTANT:")[-1].strip()
                caption = caption.replace('\n', ' ').replace('\r', ' ').replace('  ', ' ')
                
                final_caption = f"{original_caption}, {caption}" if original_caption else caption
                
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(final_caption)
                    
                print(f"  -> Original Caption: {original_caption}", flush=True)
                print(f"  -> LLaVA Output    : {caption}", flush=True)
                print(f"  -> Final Merged    : {final_caption}\n", flush=True)
            except Exception as e:
                print(f"Failed to caption {img_path.name}: {e}", flush=True)

    except Exception as e:
        print(f"Error during LLaVA execution: {e}", flush=True)
        # If it's a tenacity RetryError, unpack the actual exception
        if hasattr(e, 'last_attempt') and e.last_attempt is not None:
            print(f"Underlying exception from the final attempt: {e.last_attempt.exception()}", flush=True)
            print("Full Traceback for underlying error:", flush=True)
            traceback.print_exception(type(e.last_attempt.exception()), e.last_attempt.exception(), e.last_attempt.exception().__traceback__)
        else:
            traceback.print_exc()

    finally:
        print("Unloading LLaVA and clearing VRAM...", flush=True)
        # 100% Flush VRAM back to the system so SDXL training does not crash!
        try:
            del model
            del processor
        except NameError:
            pass
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("--- Auto-Captioning Complete ---", flush=True)
