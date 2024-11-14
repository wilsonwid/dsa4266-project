## Models

### Frame-based Convolutional Neural Networks (CNN)

Frame-based CNNs are designed for image processing, particularly in applications like computer vision. They work by applying convolutional filters to 2D frames (or images) to extract spatial features. These filters slide across the image, capturing patterns such as edges, textures, and shapes, which are then used to understand complex features through deeper layers. CNNs are particularly effective at capturing spatial hierarchies and are widely used in tasks like image classification, object detection, and facial recognition.

### Residual Networks (ResNet)

ResNet is a type of CNN architecture introduced to tackle the problem of vanishing gradients in deep networks. It achieves this through the use of “residual blocks,” which introduce shortcut connections that allow gradients to flow more directly through the network during backpropagation. This helps preserve information across layers and enables the training of very deep networks, which would otherwise suffer from performance degradation. ResNet’s architecture has been pivotal in advancing deep learning and has led to significant improvements in tasks such as image recognition and natural language processing.

### Recurrent CNNs (RCNN)

Recurrent CNNs (RCNNs) are a type of CNN architecture that applies the convolutions across different time steps in a video. This is achieved via the use of Recurrent Convolutional Layers (RCLs) that contain the convolutions previously mentioned. Theoretically, it will be able to learn spatio-temporal representations that allow it to predict better compared to normal CNNs.


### CNN-Long Short Term Memory (CNN-LSTM)

CNN-LSTM networks combine CNNs and LSTMs to handle spatiotemporal data, making them ideal for video and sequence-based tasks. The CNN component is used to extract spatial features from frames or images, while the LSTM captures the temporal dependencies across these frames. This combination allows the model to capture patterns over time, which is beneficial for tasks like action recognition in videos, human activity analysis, and sequence prediction. By leveraging both spatial and temporal information, CNN-LSTM networks achieve impressive results in understanding complex sequential data.

### Video Masked Autoencoders (VideoMAE)

[VideoMAE](https://arxiv.org/abs/2203.12602) masks random cubes and reconstructs the missing ones with an asymmetric encoder-decoder architecture. It uses an aggressive tube masking strategy with an extremely high masking ratio (90–95\%) to drop the cubes from the downsampled clips due to high temporal redundancy and correlation in videos. Thus, the model learns deeply from limited spatiotemporal data, resulting in better performance.
