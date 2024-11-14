## Models

### Frame-based Convolutional Neural Networks (CNN)

Frame-based CNNs are designed for image processing, particularly in applications like computer vision. They work by applying convolutional filters to 2D frames (or images) to extract spatial features. These filters slide across the image, capturing patterns such as edges, textures, and shapes, which are then used to understand complex features through deeper layers. CNNs are particularly effective at capturing spatial hierarchies and are widely used in tasks like image classification, object detection, and facial recognition.

### Residual Networks (ResNet)

ResNet is a type of CNN architecture introduced to tackle the problem of vanishing gradients in deep networks. It achieves this through the use of “residual blocks,” which introduce shortcut connections that allow gradients to flow more directly through the network during backpropagation. This helps preserve information across layers and enables the training of very deep networks, which would otherwise suffer from performance degradation. ResNet’s architecture has been pivotal in advancing deep learning and has led to significant improvements in tasks such as image recognition and natural language processing.

### Recurrent CNNs (RCNN)

Recurrent CNNs (RCNNs) are a type of CNN architecture that applies the convolutions across different time steps in a video. This is achieved via the use of Recurrent Convolutional Layers (RCLs) that contain the convolutions previously mentioned. Theoretically, it will be able to learn spatio-temporal representations that allow it to predict better compared to normal CNNs.

### CNN-encoder + LSTM

CNN-encoder + LSTM is a type of architecture that uses the CNN as an encoder for input to the LSTM. This CNN will be trained first on predicting classes of frames, before the classifcation head is removed and the LSTM added to the end. When training the whole model, the CNN section's weights are frozen to ensure that only the LSTM and its classification head is able to be trained.


### Video Masked Autoencoders (VideoMAE)

[VideoMAE](https://arxiv.org/abs/2203.12602) masks random cubes and reconstructs the missing ones with an asymmetric encoder-decoder architecture. It uses an aggressive tube masking strategy with an extremely high masking ratio (90–95\%) to drop the cubes from the downsampled clips due to high temporal redundancy and correlation in videos. Thus, the model learns deeply from limited spatiotemporal data, resulting in better performance.
