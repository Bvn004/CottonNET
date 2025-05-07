# 🌿 CottonNET

**CottonNET** is a Convolutional Neural Network (CNN)-based deep learning project built with **PyTorch** to classify cotton leaf images into four categories:

- Diseased Cotton Leaf
- Healthy Cotton Leaf
- Diseased Cotton Plant
- Healthy Cotton Plant

The project includes image preprocessing, data augmentation, and both a custom CNN model and a fine-tuned EfficientNet model to achieve high classification accuracy.

---

## 🌐 Try It Online

👉 [**CottonNET – Live Demo on Hugging Face Spaces**](https://huggingface.co/spaces/venkat004/CottonNET)

Upload a photo of a cotton leaf or plant to instantly get its classification!

---

## 🧠 Models Used

### 🔨 Custom CNN (from scratch)
- Built with PyTorch
- Achieved **82% accuracy**

### ⚡ EfficientNet (Pretrained)
- Fine-tuned on the same dataset
- Achieved **96% accuracy**

---

## 🔄 Data Preprocessing & Augmentation

- Performed using `torchvision.transforms`
- Includes:
  - Resizing and cropping
  - Random horizontal flips
  - Normalization
  - Random rotation

---


## 📊 Model Evaluation

- Visual and tabular evaluation of training and test performance
- Metrics used:
  - Accuracy
  - Loss plots
---

## 🚀 Getting Started

Clone the repository:

```bash
git clone https://github.com/Bvn004/CottonNET.git
cd CottonNET
