## Instructions

This model can be found on [huggingface](https://huggingface.co/shylhy/videomae-large-finetuned-deepfake-subset).
To run the trained model, simply note that this can be done with the following:
```
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
model_ckpt = "MCG-NJU/videomae-large"
model_name = model_ckpt.split("/")[-1]
new_model_name = f"shylhy/{model_name}-finetuned-deepfake-subset"
model = VideoMAEForVideoClassification.from_pretrained(new_model_name)
image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
```