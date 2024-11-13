## Instructions

This model can be found on [huggingface](https://huggingface.co/shylhy/cnn-keras-deepfake-subset).
To run the trained model, simply note that this can be done with the following:
```
from huggingface_hub import hf_hub_download
from tensorflow import keras

repo_id = "shylhy/cnn-keras-deepfake-subset"
model_file_name = "deepfake_detector_cnn.h5" 
model_file_path = hf_hub_download(repo_id=repo_id, filename=model_file_name)
model = keras.models.load_model(model_file_path)
```