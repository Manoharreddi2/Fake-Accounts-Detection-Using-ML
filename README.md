# Fake Account Detection Using Machine Learning

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



<img width="1920" height="1080" alt="Screenshot (23)" src="https://github.com/user-attachments/assets/6cfe9b7b-68fd-4b75-82c9-ee06c83ff1e3" />
<img width="1920" height="1080" alt="Screenshot (24)" src="https://github.com/user-attachments/assets/ba3da5fc-632c-4a79-9298-8ffa8eb346dc" />
<img width="1920" height="1080" alt="Screenshot (25)" src="https://github.com/user-attachments/assets/4e05d91f-97ab-4be5-9a0c-e6626439d79f" />
<img width="1920" height="1080" alt="Screenshot (26)" src="https://github.com/user-attachments/assets/6cd0139a-2559-4882-b563-d0852ce391dd" />
<img width="1920" height="1080" alt="Screenshot (27)" src="https://github.com/user-attachments/assets/6c4f3e23-b2b2-4946-a9d9-990fa9efd4ce" />
<img width="1920" height="1080" alt="Screenshot (28)" src="https://github.com/user-attachments/assets/f05d0909-04d5-4c81-b34f-3a718828a57e" />
<img width="1920" height="1080" alt="Screenshot (29)" src="https://github.com/user-attachments/assets/fc0dd9e6-7d51-4ea7-8442-73a6d113f48c" />
