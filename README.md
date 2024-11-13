# dsa4266-project

Repository for DSA4266 project.

## Setup

1. Install [pipenv](https://pypi.org/project/pipenv/):

   ```shell
   pip install pipenv
   ```

2. Create a virtual environment and install dependencies:

   ```shell
   pipenv install --dev
   ```

3. Adding dependencies:

   If you need to add a new dependency, use the following command:

   ```shell
   pipenv install <package-name>
   ```

   - Do not add the package directly to the `Pipfile`. The `Pipfile` and `Pipfile.lock` should be generated by `pipenv`.
   - You can specify if the package is a development dependency by adding the `--dev` flag.
   - Remember to commit the `Pipfile` and `Pipfile.lock` after adding a new dependency.

## Project description

The objective of this project is to develop several machine learning models capable of accurately classifying videos as either real or deepfake, which is a binary classification task. In this study, we develop several machine learning models on the Deepfake Detection Challenge (DFDC) dataset, including:

<<<<<<< HEAD
# Model Repositories
- [CNN Model](https://huggingface.co/shylhy/cnn-keras-deepfake-subset)
- [Resnet Model](https://huggingface.co/shylhy/resnet-keras-deepfake-subset)
- [VideoMAE](https://huggingface.co/shylhy/videomae-large-finetuned-deepfake-subset)

# Repository structure
=======
- Frame-based Convolutional Neural Networks (**CNN**)
- Residual Networks (**ResNet**)
- Region-based CNNs (**RCNN**)
- CNN-Long Short Term Memory (**CNN-LSTM**)
- **Vision Video Transformers**
>>>>>>> 72bb7e86ffda03bf5dc1245145914194305539b1

The models will leverage visual features extracted from video frames to distinguish deepfake videos from real videos. We then proceed to evaluate the performance of these models using the Area Under Curve (**AUC**) - Receiver Operating Characteristic (**ROC**) Curve to determine the optimal threshold to use when making predictions, before producing the classification report as well as the confusion matrix. Additionally, **high precision** and **recall** should be achieved, ensuring minimal **false positives** (incorrectly labelling real videos as fake) and **false negatives** (failing to detect a deepfake).

## Dataset

[Deepfake Detection Challenge (DFDC)](https://www.kaggle.com/competitions/deepfake-detection-challenge/data), a collaborative initiative by AWS, Facebook, Microsoft, the Partnership on AI’s Media Integrity Steering Committee, and academics.

## Instructions for Dataset

1. **Organize by Model Name**: Inside the `/data` folder, create subdirectories for each model you want to run, named exactly as specified:

- `/data/cnn`
- `/data/rcnn`
- `/data/resnet`
- `/data/videomae`
- `/data/vit`

2. **Unzip Data Files**: Unzip each model's dataset into the respective subdirectory.

3. **Verify File Paths**: Each model script should automatically refer to its dedicated folder under `/data/<model-name>`. Ensure that any required subfolders or files are placed correctly to avoid path errors.

### Preprocessed Dataset Links

- [Frames dataset](https://mega.nz/folder/fMgSib6K#kxDLFKpqvYMZSaMi3hoxCw) for frame-based models.
- [Videos dataset](https://mega.nz/file/DIBmRRgC#gDPsrAJNF4zRKA0wCj0iRbbxNl1DIuI3SRKC0AUEvoU) for video-based models.

## Repository structure

```
.
├── LICENSE
├── Pipfile
├── Pipfile.lock
├── README.md
├── __init__.py
├── data
├── eda
│   └── jh-eda.ipynb
├── evaluation
│   ├── eval_deep_fake.py
│   ├── eval_metrics.py
│   └── visualisations
│       ├── auc_roc_curve.png
│       └── confusion_matrix.png
├── example-output
├── models
│   ├── cnn
│   │   ├── cnn_dev.ipynb
│   │   ├── cnn_dev.py
│   │   ├── cnn_dev_predictions.py
│   │   └── cnn_evaluation.py
│   ├── rcnn
│   │   └── rcnn.py
│   ├── resnet
│   │   ├── Deepfake_Basic.ipynb
│   │   ├── README.md
│   │   ├── resnet_analysis.py
│   │   └── results.csv
│   ├── videomae
│   │   ├── VIDEOMAE-32frames-fp16.xlsx
│   │   └── VIDEOMAE-full-precision-24-frames.xlsx
│   └── vit
│       └── dummy.txt
├── preprocessing
│   ├── __init__.py
│   ├── augment.py
│   ├── create_balanced_dataset.ipynb
│   ├── dct.py
│   ├── dft.py
│   ├── frames.py
│   ├── label.py
│   ├── lbp.py
│   ├── split.py
│   ├── yolo.py
│   └── yolov11n-face.pt
└── utils
    ├── __init__.py
    ├── types.py
    └── utils.py
```

## Team members

In alphabetical order:

1. Au Jun Hui
1. Nixon Widjaja
1. Pong Yi Zhen
1. Sum Hung Yee
1. Tan Hui Xuan Valerie
1. Wilson Widyadhana

