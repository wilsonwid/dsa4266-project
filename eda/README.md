## Exploratory Data Analysis (EDA)

The exploratory data analysis (EDA) phase provided essential insights into the dataset's characteristics and distribution. Due to the dataset's size, the EDA was done on a smaller subset of the data to reduce computational cost and time. Visualisations were key in identifying various trends and patterns that could impact model training and performance. **Key findings** from the EDA are summarised as follows:

### Class Imbalance

A significant imbalance was observed, with a majority of the dataset consisting of deepfake videos. This imbalance may affect model accuracy and bias, necessitating strategies such as class weighting or resampling.

### Standardised Attributes

Key video attributes, including duration, frame rate, resolution, codec, and file size, were standardised across the dataset. This consistency minimises the influence of non-content-based features and helps the model focus on content-specific patterns.

### Optical Flow Analysis

Analysis of optical flow indicated that deepfake videos primarily focus on facial regions. This finding suggests that facial regions are likely to contain key features for distinguishing deepfake from real videos, making them valuable for feature extraction.

### Anomaly in Face Count

Some videos contain more than one individual, with only one being deepfaked. This anomaly will be addressed in the data preprocessing step to ensure that only relevant frames are considered for model input, improving the accuracy of predictions.
