pip install tensorflow
from tkinter import Tk, Button, Text, Scrollbar, filedialog
import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

# Global variables
filename = "fake account detection"
X, Y = None, None
X_train, X_test, y_train, y_test = None, None, None, None
model = None
accuracy = None

def loadProfileDataset():
    global filename
    global dataset
    
    filename = filedialog.askopenfilename(initialdir="Dataset", title="Select CSV File",
                                          filetypes=(("CSV Files", ".csv"), ("All Files", ".*")))
    dataset = pd.read_csv(filename)
   
    outputarea.insert(tk.END, f"{filename} loaded\n\n")
    outputarea.insert(tk.END, str(dataset.head()))

def preprocessDataset():
    global X, Y
    global dataset
    global X_train, X_test, y_train, y_test
    
    # Select features from columns (assuming last column is the target variable)
    X = dataset.iloc[:, :-1].values  # Features
    Y = dataset.iloc[:, -1].values    # Target variable (should be binary: 0 or 1)

    # Shuffle the data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    # One-hot encode the labels
    Y = to_categorical(Y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    # Output dataset information
    outputarea.insert(tk.END, f"\n\nDataset contains total Accounts: {len(X)}\n")
    outputarea.insert(tk.END, f"Total profiles used to train ANN algorithm: {len(X_train)}\n")

def executeANN():
    global model
    global X_train, X_test, y_train, y_test
    global accuracy

    # Check if the dataset has been preprocessed
    if X_train is None or X_test is None or y_train is None or y_test is None:
        outputarea.insert(tk.END, "Error: Dataset not preprocessed. Please preprocess the dataset first.\n")
        return
    
    try:
        # Define the model architecture
        model = Sequential()
        model.add(Dense(200, input_shape=(X_train.shape[1],), activation='relu', name='fc1'))
        model.add(Dense(200, activation='relu', name='fc2'))
        model.add(Dense(2, activation='softmax', name='output'))  # Binary classification (2 output nodes)

        # Compile the model
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Display model summary in the output area
        outputarea.insert(tk.END, 'ANN Neural Network Model Summary:\n')
        model.summary(print_fn=lambda x: outputarea.insert(tk.END, x + "\n"))

        # Train the model
        hist = model.fit(X_train, y_train, verbose=2, batch_size=5, epochs=50)

        # Evaluate the model
        results = model.evaluate(X_test, y_test)
        ann_acc = results[1] * 100  # Accuracy in percentage
        outputarea.insert(tk.END, f"Accuracy: {ann_acc:.2f}%\n")

        # Store accuracy history for plotting
        accuracy = hist.history
    except Exception as e:
        outputarea.insert(tk.END, f"Error during ANN execution: {str(e)}\n")

def graph():
    global accuracy
    
    acc = accuracy['accuracy']
    loss = accuracy['loss']
    
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy/Loss')
    plt.plot(acc, 'ro-', color='green')
    plt.plot(loss, 'ro-', color='blue')
    plt.legend(['Accuracy', 'Loss'], loc='upper left')
    plt.title('ANN Iteration Wise Accuracy & Loss Graph')
    plt.show()

def predictProfile():
    global model
    
    filename = filedialog.askopenfilename(initialdir="Dataset", title="Select CSV File",
                                          filetypes=(("CSV Files", ".csv"), ("All Files", ".*")))
    test = pd.read_csv(filename)
    test_features = test.iloc[:, :-1].values  # Exclude the target variable (fake column)

    # Ensure the number of features matches what the model expects
    if test_features.shape[1] != X_train.shape[1]:
        outputarea.insert(tk.END, f"Error: Expected {X_train.shape[1]} features, but got {test_features.shape[1]} features.\n")
        return

    # Make predictions
    predictions = model.predict(test_features)
    predicted_classes = np.argmax(predictions, axis=1)

    for i in range(len(test_features)):
        # Format the account details neatly
        account_details = ", ".join([f"{col}: {value}" for col, value in zip(test.columns[:-1], test_features[i])])
        
        if predicted_classes[i] == 0:
            emoji = "✅"  # Checkmark for genuine
            msg = "Account Details Predicted As Genuine"
        else:
            emoji = "❌"  # Cross for fake
            msg = "Account Details Predicted As Fake"
        
        # Insert formatted output into the output area
        outputarea.insert(tk.END, f"{account_details} | {emoji} {msg}\n\n")

def close():
    main.destroy()

# Initialize the main window
main = Tk()
main.title("Fake Account Detection Using Machine Learning and Data Science")
main.geometry("1300x1200")
main.config(bg="lightgreen")

font = ('times', 15, 'bold')
title = tk.Label(main, text='Fake Account Detection Using Machine Learning and Data Science', font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

ff = ('times', 12, 'bold')
buttons = [
    ("Upload Social Network Profiles Dataset", loadProfileDataset, 100),
    ("Preprocess Dataset", preprocessDataset, 150),
    ("Run ANN Algorithm", executeANN, 200),
    ("ANN Accuracy & Loss Graph", graph, 250),
    ("Predict Fake/Genuine Profile using ANN", predictProfile, 300),
    ("Logout", close, 350)
]

for text, command, y_pos in buttons:
    btn = Button(main, text=text, command=command, font=ff)
    btn.place(x=20, y=y_pos)

font1 = ('times', 12, 'bold')
outputarea = Text(main, height=30, width=85)
scroll = Scrollbar(outputarea)
outputarea.configure(yscrollcommand=scroll.set)
outputarea.place(x=400, y=100)
outputarea.config(font=font1)

main.mainloop()
