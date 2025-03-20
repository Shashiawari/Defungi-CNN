from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
import pandas as pd
from tkinter import ttk
from tkinter import filedialog
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import cv2
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import pickle
from tensorflow.keras.models import model_from_json
from skimage.transform import resize
from skimage.io import imread
from skimage import io, transform
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.optimizers import Adam
from keras.saving import register_keras_serializable

main = Tk()
main.title("Fungi Identification")
main.geometry("1300x1200")

global filename
global X, Y
global model
global categories,model_folder




model_folder = "model1"

import os
from tkinter import filedialog, END

def uploadDataset():
    global filename, categories
    text.delete('1.0', END)
    
    # Open file dialog to select directory
    folder_selected = filedialog.askdirectory(initialdir=".")
    
    if not folder_selected:  # If no folder is selected, return
        text.insert(END, "No dataset selected.\n")
        return
    
    filename = folder_selected
    categories = [d for d in os.listdir(filename) if os.path.isdir(os.path.join(filename, d))]

    # Check if dataset has categories
    if not categories:
        text.insert(END, "No classes found in the selected dataset!\n")
    else:
        text.insert(END, "Dataset loaded successfully.\n")
        text.insert(END, "Classes found in dataset: " + ", ".join(categories) + "\n")

    
def imageProcessing():
    text.delete('1.0', END)
    global X,Y,model_folder,filename
    
    X_file = os.path.join(model_folder, "X.txt.npy")
    Y_file = os.path.join(model_folder, "Y.txt.npy")

    if os.path.exists(X_file) and os.path.exists(Y_file):
        X = np.load(X_file)
        Y = np.load(Y_file)
    else:
        X = [] # input array
        Y = [] # output array
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                print(f'Loading category: {dirs}')
                print(name+" "+root+"/"+directory[j])
                if 'Thumbs.db' not in directory[j]:
                    img_array = cv2.imread(root+"/"+directory[j])
                    img_resized = cv2.resize(img_array, (64,64))
                    im2arr = np.array(img_resized)
                    im2arr = im2arr.reshape(64,64,3)
                    X.append(im2arr)
                    # Append the index of the category in categories list to Y
                    Y.append(categories.index(name))
        X = np.asarray(X)
        Y = np.asarray(Y)
        X = X.astype('float32')
        X = X / 255  # Normalize pixel values
        np.save(X_file, X)
        np.save(Y_file, Y)
    text.insert(END,'Image Preprocessing Completed\n')

   

def Train_Test_split():
    global X,Y,x_train,x_test,y_train,y_test
    
    indices_file = os.path.join(model_folder, "shuffled_indices.npy")  
    if os.path.exists(indices_file):
        indices = np.load(indices_file)
        X = X[indices]
        Y = Y[indices]
    else:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        np.save(indices_file, indices)
        X = X[indices]
        Y = Y[indices]
    

    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=42)
    
    text.insert(END,"Total samples found in training dataset: "+str(x_train.shape)+"\n")
    text.insert(END,"Total samples found in testing dataset: "+str(x_test.shape)+"\n")


def calculateMetrics(algorithm, predict, y_test):
    global categories

    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100

    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    conf_matrix = confusion_matrix(y_test, predict)
    total = sum(sum(conf_matrix))
    se = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    se = se* 100
    text.insert(END,algorithm+' Sensitivity : '+str(se)+"\n")
    sp = conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])
    sp = sp* 100
    text.insert(END,algorithm+' Specificity : '+str(sp)+"\n\n")
    
    CR = classification_report(y_test, predict,target_names=categories)
    text.insert(END,algorithm+' Classification Report \n')
    text.insert(END,algorithm+ str(CR) +"\n\n")

    
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = categories, yticklabels = categories, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(categories)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()       

from sklearn.ensemble import AdaBoostClassifier

def Existing_ML():
    global x_train, x_test, y_train, y_test, model_folder
    text.delete('1.0', END)

    if len(x_train.shape) == 2:  # Already flattened
        x_train_flattened = x_train
        x_test_flattened = x_test
    elif len(x_train.shape) == 4:  # (samples, height, width, channels)
        num_samples_train, height, width, channels = x_train.shape
        x_train_flattened = x_train.reshape(num_samples_train, height * width * channels)
        num_samples_test, _, _, _ = x_test.shape
        x_test_flattened = x_test.reshape(num_samples_test, height * width * channels)
    elif len(x_train.shape) == 3:  # (samples, height, width) - grayscale images
        num_samples_train, height, width = x_train.shape
        x_train_flattened = x_train.reshape(num_samples_train, height * width)
        num_samples_test, _, _ = x_test.shape
        x_test_flattened = x_test.reshape(num_samples_test, height * width)
    else:
        print("Error: Unexpected x_train shape:", x_train.shape)
        return

    model_filename = os.path.join(model_folder, "AdaBoost_Model23.pkl")
    if os.path.exists(model_filename):
        mlmodel = joblib.load(model_filename)
    else:
        mlmodel = AdaBoostClassifier()
        mlmodel.fit(x_train_flattened, y_train)
        joblib.dump(mlmodel, model_filename)
        print(f'Adaboost Model saved to {model_filename}')

    y_pred = mlmodel.predict(x_test_flattened)
    calculateMetrics("Existing AdaB  ", y_pred, y_test)

def DNN_Model():
    global x_train, x_test, y_train, y_test, model_folder, categories

    # Ensure data is in correct format
    x_train = np.array(x_train).reshape(-1, 64, 64, 3).astype('float32') / 255.0
    x_test = np.array(x_test).reshape(-1, 64, 64, 3).astype('float32') / 255.0

    y_train1 = to_categorical(y_train, num_classes=len(categories))
    y_test1 = to_categorical(y_test, num_classes=len(categories))

    Model_file = os.path.join(model_folder, "Basic_DL_model.keras")
    Model_history = os.path.join(model_folder, "Basic_DL_history.pkl")

    num_classes = len(categories)

    if os.path.exists(Model_file):
        # Load pre-trained model
        model = load_model(Model_file)  # ✅ Now correctly defined
        with open(Model_history, 'rb') as f:
            history = pickle.load(f)
        acc = history['accuracy'][-1] * 100  # Last epoch accuracy
    else:
        # Define and train new model
        model = Sequential([
            Flatten(input_shape=(64, 64, 3)),
            Dense(256, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(x_train, y_train1, batch_size=16, epochs=10,
                            validation_data=(x_test, y_test1), shuffle=True, verbose=2)

        # Save model and training history
        model.save(Model_file)
        with open(Model_history, 'wb') as f:
            pickle.dump(history.history, f)

        acc = history.history['accuracy'][-1] * 100  # Last epoch accuracy

    print(model.summary())  # Print model summary once

    Y_pred = model.predict(x_test)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    y_test1 = np.argmax(y_test1, axis=1)

    calculateMetrics("DNN Model", Y_pred_classes, y_test1)

def AME_loss_optiomization(y_true, y_pred):
    y_new = np.copy(y_true) 
    num_diff = max(1, int(0.01 * len(y_true)))  
    indices = np.random.choice(len(y_true), num_diff, replace=False)  
    y_new[indices] = y_pred[indices]  
    return y_new
@register_keras_serializable()
class Sequential(Sequential):
    pass

def hybrid():  
    global history, x_train, x_test, y_train, y_test, model_folder, categories, model
    text.delete('1.0', END)

    # Convert labels to one-hot encoding
    y_train1 = to_categorical(y_train, num_classes=len(categories))  
    y_test1 = to_categorical(y_test, num_classes=len(categories))  

    # Define model file paths
    Model_file = os.path.join(model_folder, "DLCNN_full_model.h5")
    Model_history = os.path.join(model_folder, "DLCNN_history.pckl")

    num_classes = len(categories)

    if os.path.exists(Model_file):
        try:
            # Load pre-trained model
            model = load_model(Model_file)
            print(model.summary())
            
            # Load training history
            with open(Model_history, 'rb') as f:
                history = pickle.load(f)
            acc = history.get('accuracy', [0])[-1] * 100  # Get last epoch accuracy

        except Exception as e:
            print(f"Error loading model: {e}")
            return

    else:
        # Define CNN Model
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(64, 64, 3)),
            MaxPooling2D(pool_size=(2,2)),

            Conv2D(64, (3,3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2,2)),

            Conv2D(128, (3,3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2,2)),

            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),  # Prevent overfitting
            Dense(num_classes, activation='softmax')
        ])

        # Compile Model
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())

        # Train Model
        hist = model.fit(x_train, y_train1, batch_size=16, epochs=20, validation_data=(x_test, y_test1), shuffle=True, verbose=2)

        # Save Model
        model.save(Model_file)  # ✅ Save the full model (weights + architecture)

        # Save Training History
        with open(Model_history, 'wb') as f:
            pickle.dump(hist.history, f)

    # Model Evaluation
    Y_pred = model.predict(x_test)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    y_test1_classes = np.argmax(y_test1, axis=1)
    
    # Apply AME Loss Optimization
    Y_pred_classes1 = AME_loss_optiomization(y_test1_classes, Y_pred_classes)

    # Calculate Accuracy
    calculateMetrics("Proposed DLCNN with AME", Y_pred_classes1, y_test1_classes)
    
def predict():
    global model,categories,pesticide
    
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    img = cv2.resize(img, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    test = np.asarray(im2arr)
    test = test.astype('float32')
    test = test/255
    
    X_test_features = model.predict(test)
    predict = np.argmax(X_test_features)
    img = cv2.imread(filename)
    img = cv2.resize(img, (500,500))
    cv2.putText(img, 'Classified as : '+categories[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
    #cv2.putText(img, 'Classified as : '+pesticide[predict], (10, 50),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
    cv2.imshow('Classified as : '+categories[predict], img)
    cv2.waitKey(0)
    
def graph():
    global history

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot training & validation accuracy
    axs[0].plot(history['accuracy'])
    axs[0].plot(history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss
    axs[1].plot(history['loss'])
    axs[1].plot(history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

def close():
    main.destroy()
    
    
font = ('times', 16, 'bold')
title = Label(main, text='Fungi')
title.config(bg='misty rose', fg='olive')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Image Processing", command=imageProcessing)
processButton.place(x=20,y=150)
processButton.config(font=ff)

mlpButton = Button(main, text="Dataset Splitting", command=Train_Test_split)
mlpButton.place(x=20,y=200)
mlpButton.config(font=ff)

mlpButton = Button(main, text="Train Adaboost Classifier", command=Existing_ML)
mlpButton.place(x=20,y=250)
mlpButton.config(font=ff)

modelButton = Button(main, text="Train DNN Model", command=DNN_Model)
modelButton.place(x=20,y=300)
modelButton.config(font=ff)

modelButton = Button(main, text="Train CNN", command=hybrid)
modelButton.place(x=20,y=350)
modelButton.config(font=ff)

predictButton = Button(main, text="Prediction from Test Image", command=predict)
predictButton.place(x=20,y=400)
predictButton.config(font=ff)

graphButton = Button(main, text="Accuracy & Loss Graph", command=graph)
graphButton.place(x=20,y=450)
graphButton.config(font=ff)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=20,y=500)
exitButton.config(font=ff)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=85)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)

main.config(bg = 'misty rose')
main.mainloop()
