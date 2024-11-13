## Models

### Frame-based Convolutional Neural Networks (CNN)

Frame-based CNNs are designed for image processing, particularly in applications like computer vision. They work by applying convolutional filters to 2D frames (or images) to extract spatial features. These filters slide across the image, capturing patterns such as edges, textures, and shapes, which are then used to understand complex features through deeper layers. CNNs are particularly effective at capturing spatial hierarchies and are widely used in tasks like image classification, object detection, and facial recognition.

### Residual Networks (ResNet)

ResNet is a type of CNN architecture introduced to tackle the problem of vanishing gradients in deep networks. It achieves this through the use of “residual blocks,” which introduce shortcut connections that allow gradients to flow more directly through the network during backpropagation. This helps preserve information across layers and enables the training of very deep networks, which would otherwise suffer from performance degradation. ResNet’s architecture has been pivotal in advancing deep learning and has led to significant improvements in tasks such as image recognition and natural language processing.

### Region-based CNNs (RCNN)

RCNNs are specialized CNNs used for object detection, where the task is to identify and localize objects within an image. Unlike traditional CNNs, RCNNs first generate region proposals, which are likely areas containing objects, and then use a CNN to classify these regions. This approach allows RCNNs to efficiently focus on object detection within complex scenes, making them useful for applications like autonomous driving, surveillance, and image analysis. Variants like Fast RCNN and Faster RCNN have been developed to improve processing speed and accuracy.

### CNN-Long Short Term Memory (CNN-LSTM)

CNN-LSTM networks combine CNNs and LSTMs to handle spatiotemporal data, making them ideal for video and sequence-based tasks. The CNN component is used to extract spatial features from frames or images, while the LSTM captures the temporal dependencies across these frames. This combination allows the model to capture patterns over time, which is beneficial for tasks like action recognition in videos, human activity analysis, and sequence prediction. By leveraging both spatial and temporal information, CNN-LSTM networks achieve impressive results in understanding complex sequential data.

### Vision Video Transformers

Vision Video Transformers (ViViT) are transformer-based models specifically designed for video understanding. Unlike CNNs, transformers can capture long-range dependencies and context, making them particularly suited for temporal data. ViViT processes both spatial and temporal information within videos by treating the sequence of video frames as tokens and applying self-attention mechanisms to analyze relationships between these tokens. This approach is effective in tasks like action recognition, event detection, and video classification. Vision transformers are highly flexible and have shown promise in surpassing CNNs in certain vision tasks by leveraging the transformer’s attention mechanism.
