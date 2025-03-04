import os
import time
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_LOGS"] = "+recompiles"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import cv2
import hydra
import torch
from omegaconf import OmegaConf
from torchvision.ops import box_convert

from dataloader import video_loader, inv_normalize
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.inference import load_model, annotate

def annotate_frame(image, prompt, thresh, logits, boxes, tokenizer):
    tokens = tokenizer(prompt)
    phrases = [
            get_phrases_from_posmap(logit > thresh, tokens, tokenizer).replace('.', '')
            for logit
            in logits
        ]
    logits = logits.max(dim=1)[0]
    
    return annotate(image_source=image, boxes=boxes, logits=logits, phrases=phrases)
    

def inference(config):
    dataloader = video_loader(config)
    model = load_model(config.model.dino_config, config.model.weights).to(config.device)
    prompt = [config.model.prompt] * config.batch_size
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            start = time.time()
            frames, creator_id, video_id = data
            creator_id = creator_id[0]
            video_id = video_id[0]
            frames = frames[0].to(device=config.device, dtype=torch.float32)
            outputs = model(frames, captions=prompt)
            print(f"Processing {i}th video: {creator_id}_{video_id}")
            
            prediction_logits = outputs["pred_logits"].cpu().sigmoid()
            prediction_boxes = outputs["pred_boxes"].cpu()
            box_mask = prediction_logits.max(dim=2)[0] > config.model.box_threshold
            
            for j in range(config.batch_size):
                boxes = prediction_boxes[j][box_mask[j]]
                logits = prediction_logits[j][box_mask[j]]
                
                if boxes.numel() > 0:
                    frame = annotate_frame(
                        image=inv_normalize(frames[j]),
                        prompt=config.model.prompt,
                        thresh=config.model.text_threshold,
                        logits=logits,
                        boxes=boxes,
                        tokenizer=model.tokenizer
                    )
                    
                    # Save the annotated frame
                    if not os.path.exists(os.path.join(config.out_dir, creator_id, video_id)):
                        os.makedirs(os.path.join(config.out_dir, creator_id, video_id))
                    
                    cv2.imwrite(
                        os.path.join(config.out_dir, creator_id, video_id, f"{j}.jpg"),
                        frame
                    )
                
                # TODO: Extract patches
                # xyxys = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
                
                # rectangular scaling
                # xyxys[:, 0] *= config.xraw
                # xyxys[:, 1] *= config.yraw
                # xyxys[:, 2] *= config.xraw
                # xyxys[:, 3] *= config.yraw
            
            # breakpoint()
            print(f"Processed {i}th video in {time.time() - start:.2f} seconds")


@hydra.main(version_base=None, config_path="configs", config_name="dino.yaml")
def main(config):
    OmegaConf.resolve(config)
    inference(config)

    
if __name__ == "__main__":
    main()