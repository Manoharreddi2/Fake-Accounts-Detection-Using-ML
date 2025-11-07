# Fake Account Detection Using Machine Learning and Data Science

This project uses an Artificial Neural Network (ANN) to classify social network profiles as **Fake** or **Genuine** based on user-provided dataset features. The application provides an easy-to-use **Graphical User Interface (GUI)** built with **Tkinter**, allowing users to upload datasets, preprocess data, train an ANN model, visualize training accuracy and loss, and predict new profile categories.

---

## 🚀 Features

| Feature | Description |
|--------|-------------|
| **Dataset Upload** | User can upload CSV dataset containing social network profile features. |
| **Preprocessing** | Data is shuffled, one-hot encoded, and split into train/test sets. |
| **ANN Model Training** | A 3-layer ANN model is trained using Keras + TensorFlow backend. |
| **Model Evaluation** | Displays accuracy of the trained model. |
| **Graph Visualization** | Plots training accuracy and loss curves. |
| **Profile Prediction** | Classifies new profiles as **Fake** or **Genuine**. |
| **Simple GUI** | Provides easy button-based control for all functions. |

---

## 📂 Dataset Format

The dataset must be in **CSV** format.

- **All columns except the last column** are treated as input features.
- **Last column** should be the **label**:
