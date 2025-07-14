🌿 Edge AI Image Classification with TensorFlow Lite
Lightweight image classification model built for Edge AI deployment, using the TF Flowers dataset and TensorFlow Lite. Ideal for real-time, privacy-preserving applications on low-power devices.

🚀 Project Overview
This project demonstrates how to:

Train a convolutional neural network using MobileNetV2

Preprocess and load data with TensorFlow Datasets (TFDS)

Convert a trained model to TensorFlow Lite format for deployment on edge devices

Highlight the benefits of Edge AI in latency, privacy, and offline accessibility

📊 Dataset Used
TF Flowers

✅ 3,670+ flower images

🏷️ 5 classes

📦 TensorFlow Datasets - tf_flowers

🧠 Model Architecture
Base Model: MobileNetV2 (pretrained on ImageNet)

Custom Layers:

GlobalAveragePooling2D

Dense(128, ReLU)

Dense(output_classes, Softmax)

Final Accuracy: ~92% on validation set

📦 TensorFlow Lite Conversion
After training, the model is:

✅ Saved in .h5 format

✅ Converted to .tflite using TFLiteConverter

✅ Ready for deployment on Raspberry Pi, Android, or IoT microcontrollers

🖥️ Edge AI Deployment Flow
mermaid
Copy
Edit
graph LR
A[Image Input] --> B[Model Training]
B --> C[TFLite Conversion]
C --> D[Deployed to Edge Device]
💡 Edge AI Benefits:

🔄 Real-time predictions

🔐 Data stays on-device

🌐 Works offline

🛠️ How to Run
Colab:
Upload and run edge_ai_model.ipynb in Google Colab

Local Python:
Clone this repo and run:

bash
Copy
Edit
pip install tensorflow tensorflow-datasets
python edge_ai_model.py
To Convert Model to TFLite:

python
Copy
Edit
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
📁 Folder Structure
bash
Copy
Edit
EdgeAI-Flowers/
│
├── edge_ai_model.ipynb             # Notebook version (Colab-ready)
├── edge_ai_model.py                # Python script version
├── flower_classifier_model.h5      # Saved Keras model
├── flower_classifier_model.tflite  # TFLite optimized model
├── README.md                       # This file
└── edge_ai_deployment_flow.png     # Deployment architecture diagram
📜 License
MIT License — use, modify, and deploy freely 🌱
