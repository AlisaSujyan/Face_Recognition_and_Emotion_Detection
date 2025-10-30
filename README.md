# Facial Emotion Recognition System

A comprehensive computer vision project combining **face recognition** and **emotion detection** using deep learning techniques. This project explores state-of-the-art approaches for identifying faces and classifying emotional expressions from facial images.

---


##  Project Overview

This project implements a dual-purpose facial analysis system:

1. **Face Recognition**: Identify individuals from facial images using deep face embeddings
2. **Emotion Detection**: Classify facial expressions into 7 emotion categories

### Key Features

-  Face detection and alignment using MTCNN/Haar Cascade
-  Face recognition using embedding-based approaches (FaceNet/ArcFace)
-  Emotion classification with CNN architectures
-  Comprehensive data exploration and preprocessing pipelines
-  Handling class imbalance with weighted loss functions
-  Extensive data augmentation techniques


---

##  Datasets

### 1. LFW (Labeled Faces in the Wild) - Face Recognition

**Source**: University of Massachusetts, Amherst  
**Version**: lfw-deepfunneled (aligned faces)  
**Link**: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset

**Statistics**:
- **Total Images**: ~13,000
- **Unique Identities**: 5,749+
- **Image Format**: RGB, varying sizes (typically ~250×250)
- **Challenge**: Unconstrained real-world conditions

**Characteristics**:
- Severe class imbalance (most people have 1-5 images)
- Wide variation in pose, lighting, and expression
- Real-world quality variations
- Pre-aligned faces for better feature extraction

**Preprocessing Applied**:
- Filtered to identities with ≥10 images
- Face detection and cropping with margin
- Resized to 160×160 pixels (FaceNet standard)
- Normalized to [-1, 1] range
- Train/Val/Test split: 70%/15%/15%

---

### 2. FER2013 (Facial Expression Recognition 2013) - Emotion Detection

**Source**: Kaggle Challenge 2013  
**Link**: https://www.kaggle.com/datasets/msambare/fer2013

**Statistics**:
- **Total Images**: ~35,000
- **Train Set**: ~28,000 images
- **Test Set**: ~7,000 images
- **Image Size**: 48×48 pixels (grayscale)
- **Classes**: 7 emotions

**Emotion Classes**:
1. **Angry** 
2. **Disgust** 
3. **Fear** 
4. **Happy** 
5. **Sad** 
6. **Surprise** 
7. **Neutral** 

**Challenges**:
- Low image resolution (48×48)
- Class imbalance (disgust class has significantly fewer samples)
- Label noise (~8-10% mislabeled according to literature)
- Grayscale images only

**Preprocessing Applied**:
- Normalized to [0, 1] range
- Heavy data augmentation (rotation, flipping, zoom, brightness, contrast, noise)
- Class weight calculation for handling imbalance
- Train/Val split: 85%/15%

---



**Models for face recognition**:

1. **DeepFace (Facebook, 2014)**
   - First CNN achieving near-human accuracy
   - 97.35% accuracy on LFW
   - 3D face alignment + deep CNN

2. **FaceNet (Google, 2015)**
   - Triplet loss for learning embeddings
   - 128/512-dimensional embeddings
   - 99.63% accuracy on LFW
   - **Architecture**: Inception-ResNet-v1

3. **VGGFace/VGGFace2 (Oxford, 2015/2018)**
   - Large-scale training data (2.6M/3.3M images)
   - VGG-16 and ResNet-50 architectures
   - Excellent pre-trained models

4. **ArcFace (2019) - Current SOTA**
   - Additive Angular Margin Loss
   - Better feature discrimination
   - 99.83% accuracy on LFW
   - Superior to softmax and triplet loss


**My Approach**:
- Use pre-trained FaceNet/ArcFace for embedding extraction
- Fine-tune on LFW for specific identities

---

**Methods for emotion detection**:


1. **Baseline CNN**
   ```
   Conv2D(32) → Pool → Conv2D(64) → Pool → Conv2D(128) → Pool → FC(512) → FC(7)
   ```
   - Simple but effective for FER2013
   - ~65-70% accuracy

2. **VGG-Style Networks**
   - Deeper architecture (16-19 layers)
   - Small 3×3 filters
   - ~70-73% accuracy on FER2013

3. **ResNet (Residual Networks)**
   - Skip connections prevent vanishing gradients
   - Deeper networks (50-101 layers)
   - Better feature extraction

4. **Attention Mechanisms**
   - Focus on relevant facial regions (eyes, mouth)
   - Spatial attention + channel attention
   - Improved interpretability

**Transfer Learning**:
- Pre-train on ImageNet or face datasets
- Fine-tune on emotion datasets

**Current State-of-the-Art on FER2013**:
- **Ensemble Methods**: 73-75% accuracy
- **Attention Networks**: 72-73% accuracy
- **Transfer Learning + Fine-tuning**: 70-72% accuracy

**My Approach**:
- Start with custom CNN baseline
- Experiment with transfer learning (VGG16/ResNet50)
- Apply heavy data augmentation
- Use class weights to handle imbalance

---
