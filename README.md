### 🦴 Bone Fracture Classification using Deep Learning

This project is a web application built with **Streamlit** that detects **bone fractures** from X-ray images using a trained deep learning model. The model classifies whether a bone is **fractured** or **not fractured**.

---

### 🚀 Try the App

👉 [Click here to open the Bone Fracture Detection App](https://bonefractureclassification-vgqukayvd8kipm4ppnrvqh.streamlit.app/)

Upload an X-ray image, and the app will analyze and classify it using a pre-trained neural network.

---

### 🧠 Model Information

The model was trained using a Convolutional Neural Network (CNN) on a labeled dataset of multi-region bone X-ray images. It predicts whether the input image shows signs of a fracture.

---

### 📁 Dataset

* **Source:** Kaggle
* **Name:** [Fracture multi-region X-ray data](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data)
* The dataset includes both fractured and non-fractured X-ray images across various body regions.

---

### 📦 Installation

To run this project locally:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/bonefractureclassification.git
   cd bonefractureclassification
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

---

### 💡 Features

* Simple UI for uploading and predicting fracture presence.
* Real-time model inference.
* Works on a variety of X-ray images (leg, hand, etc.).

---

### 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* Streamlit
* NumPy, PIL
