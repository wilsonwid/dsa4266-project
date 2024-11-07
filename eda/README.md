## Exploratory Data Analysis (EDA)

The exploratory data analysis (EDA) phase provided essential insights into the dataset's characteristics and distribution. Due to the dataset's size, the EDA was done on a smaller subset of the data to reduce computational cost and time. Visualisations were key in identifying various trends and patterns that could impact model training and performance.

## EDA Techniques Implemented

1. **Data Loading and Inspection**

   - Utilizes `pandas` to load the dataset and inspect its structure.
   - Initial inspection of data types, missing values, and overall shape of the dataset.

2. **Statistical Summary**

   - Displays descriptive statistics (mean, median, standard deviation, etc.) for numerical columns to understand central tendency and spread.

3. **Data Visualization**

   - **Histograms**: Used for visualizing distributions of key features to observe any skewness or outliers.
   - **Box Plots**: Created to detect outliers and understand the range of feature values.
   - **Scatter Plots**: Examined feature relationships and potential patterns.
   - **3D Visualization**: Used `plotly` for interactive 3D visualization to explore multi-dimensional feature space.

4. **Dimensionality Reduction**

   - **PCA (Principal Component Analysis)**: Implemented to reduce the dataset to principal components, aiding in visualizing the primary directions of variance.
   - **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Used for high-dimensional data visualization by projecting it onto a lower-dimensional space, enhancing the understanding of clusters or separable groups.

5. **Image and Video Data Handling**
   - Processed media data using libraries like `OpenCV` and `ffmpeg`.
   - Utilized PyTorch's `torchvision` for image transformations relevant to deepfake media content.

## Libraries Used

- **Data Processing**: `pandas`, `numpy`, `json`, `os`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Dimensionality Reduction**: `scikit-learn` (PCA, t-SNE)
- **Media Processing**: `OpenCV`, `ffmpeg`, `torch`, `torchvision`

**Key findings** from the EDA are summarised as follows:

### Class Imbalance

A significant imbalance was observed, with a majority of the dataset consisting of deepfake videos. This imbalance may affect model accuracy and bias, necessitating strategies such as class weighting or resampling.

### Standardised Attributes

Key video attributes, including duration, frame rate, resolution, codec, and file size, were standardised across the dataset. This consistency minimises the influence of non-content-based features and helps the model focus on content-specific patterns.

### Optical Flow Analysis

Analysis of optical flow indicated that deepfake videos primarily focus on facial regions. This finding suggests that facial regions are likely to contain key features for distinguishing deepfake from real videos, making them valuable for feature extraction.

### Anomaly in Face Count

Some videos contain more than one individual, with only one being deepfaked. This anomaly will be addressed in the data preprocessing step to ensure that only relevant frames are considered for model input, improving the accuracy of predictions.
