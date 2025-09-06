# 🍔👁 Food Vision Big™ — Milestone Project 1

This repository contains **Milestone Project 1** from the *Food Vision* series, where we build, train, and evaluate deep learning models on a large-scale **food image classification problem**. Using **TensorFlow 2.x** and **Keras**, the project demonstrates how to leverage **transfer learning**, **mixed precision training**, and **TensorBoard logging** to achieve high performance on image datasets.

---

## 📌 Project Overview
The challenge: **classify food images into multiple categories** with high accuracy.  
This project extends previous Food Vision experiments into a **bigger dataset** (tens of thousands of images) to test the scalability and generalization of deep learning models.  

Key aspects of the project:
- **Data pipeline** built with `tensorflow_datasets`  
- **Pre-trained CNN backbones** (transfer learning)  
- **Mixed precision training** for improved speed and efficiency  
- **Custom callbacks** for monitoring and early stopping  
- **TensorBoard logs** for real-time insights  
- **Performance visualization** and **model comparison**  

---

## ⚙️ Installation & Requirements

Clone this repository and install dependencies:

```bash
git clone https://github.com/<your-username>/food-vision-big.git
cd food-vision-big
pip install -r requirements.txt
```

### Requirements
Main libraries used:
- `tensorflow` (Deep Learning framework)
- `tensorflow_datasets` (Dataset loader)
- `matplotlib` (Visualizations)
- `numpy` (Numerical computations)
- `os`, `datetime` (Utilities)

---

## 📊 Dataset
- Source: `tensorflow_datasets` **Food101 dataset**  
- **101 food categories**, ~101,000 images  
- Pre-split into **training** and **validation** sets  
- Images resized and normalized for consistency  
- Supports batching and prefetching for efficient GPU training  

---

## 🧠 Methodology
1. **Data Preprocessing**  
   - Normalize pixel values  
   - Resize images to uniform dimensions  
   - Create efficient pipelines with `tf.data`  

2. **Model Building**  
   - Use pre-trained CNN backbones (e.g., EfficientNet, ResNet)  
   - Replace top layers with dense classification head  
   - Apply dropout for regularization  

3. **Training**  
   - Loss function: `SparseCategoricalCrossentropy`  
   - Optimizer: `Adam` with learning rate scheduling  
   - Mixed precision enabled for GPU acceleration  
   - Callbacks:  
     - `create_tensorboard_callback` → log metrics  
     - `EarlyStopping` → prevent overfitting  

4. **Evaluation & Visualization**  
   - `plot_loss_curves`: visualize accuracy/loss trends  
   - `compare_historys`: compare fine-tuning runs  
   - TensorBoard scalars & histograms  

---

## ⚡ Mixed Precision Training

Mixed precision uses **both 16-bit and 32-bit floating point types**
during training to speed up computation and reduce memory usage, while
maintaining model accuracy.

-   Enabled via:

    ``` python
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    ```

### 🔧 GPU Compatibility

Mixed precision training is **hardware-dependent**.\
- Best supported on **NVIDIA GPUs with Tensor Cores** (Volta, Turing,
Ampere, or newer).\
- Examples:\
- Tesla V100, T4\
- RTX 20xx, RTX 30xx, RTX 40xx series\
- A100, H100 data center GPUs\
- On unsupported GPUs/CPUs, TensorFlow will automatically fall back to
standard float32 precision.

### ✅ Benefits

-   Faster training on compatible GPUs\
-   Reduced memory footprint → allows larger batch sizes\
-   No significant drop in accuracy

In this project, mixed precision reduced training time by **\~40%**
while maintaining accuracy, making it a critical optimization for
scaling deep learning experiments.


---

## 📈 Results & Insights

We trained models on the **Food101 dataset (101 food categories, >100k images)** using transfer learning with **EfficientNetB0**.  

### 🔹 Phase 1: Feature Extraction
- **Training setup**: 3 epochs  
- **Final results**:  
  - Training Accuracy: **72.4%**  
  - Validation Accuracy: **72.4%**  
  - Training Loss: **1.05**  
  - Validation Loss: **0.99**  

### 🔹 Phase 2: Fine-Tuning
- **Training setup**: Unfroze top layers of EfficientNetB0, trained for multiple epochs with learning rate scheduling  
- **Best results**:  
  - Training Accuracy: **99.2%**  
  - Validation Accuracy: **80.3%**  
  - Validation Loss: **1.05**  


---

## 🔮 Future Work
- Try larger architectures (EfficientNetB7, Vision Transformers)  
- Apply **data augmentation** for robustness  
- Deploy as a **Flask/FastAPI web app** or TensorFlow Serving model  
- Experiment with **hyperparameter tuning** (learning rates, optimizers)  

