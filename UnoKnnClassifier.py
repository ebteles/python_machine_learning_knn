from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
import pickle


def image_to_feature_vector(image, size=(32, 32)):
    return cv2.resize(image, size).flatten()


def extract_color_histogram(image, bins=(8, 8, 8)):
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])

    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    else:
        cv2.normalize(hist, hist)

    return hist.flatten()


def model_load(file_name):
    
    model = pickle.load(open(file_name, 'rb'))
    classes_file = file_name[:-10] + '_labels.txt'
    classes = []
    with open(classes_file) as f:
        classes = [row.strip() for row in f]
    return model, classes


def predict_rpi(model, classes, path_image='', image=None):
    
    if path_image != '':
        img = [image_to_feature_vector(cv2.imread(path_image))]
    else:
        img = [image_to_feature_vector(image)]

    classe  = model.predict(img)[0]
    id_classe = classes.index(classe)
    probabilidade= model.predict_proba(img)[0]
    
    return classe, probabilidade[id_classe] * 100



def predict_his(model, classes, path_image='', image=None):
    
    if path_image != '':
        img = cv2.imread(path_image)
    else:
        img = image
    
    img = [extract_color_histogram(img)]

    classe  = model.predict(img)[0]
    id_classe = classes.index(classe)

    probabilidade= model.predict_proba(img)[0]
    
    return classe, probabilidade[id_classe] * 100



def train_knn(model_name, num_neighbors, num_jobs, train_img, train_lbl, test_img, test_lbl):
    
    model = KNeighborsClassifier(n_neighbors=num_neighbors, n_jobs=num_jobs)
    model.fit(train_img, train_lbl)
    
    acc = model.score(test_img, test_lbl)
    
    # salvar
    pickle.dump(model, open(model_name, 'wb'))

    return acc * 100

def train(images_path, model_name_path, num_neighbors=1, num_jobs=1, teste_size=0.25, verbose=False):
    """
    images_path....: Lista contendo uma ou mais imagens (path + imagem)
    model_name_path: Nome do modelo (path + nome). Nao colocar extensao!!!
    """
    raw_images = []
    features = []
    labels = []
    
    name_model_rpi = model_name_path + '_rpi.model'  # raw pixel intensite
    name_model_his = model_name_path + '_his.model'  # histograma
    name_classes =  model_name_path + '_labels.txt'
    
    # loop para tratar cada uma das imagens
    for (i, image_path) in enumerate(images_path):
        
        image = cv2.imread(image_path)

        # Assumir que o classe é o nome da pasta que contem as imagens
        label = os.path.normpath(image_path).split('\\')[-2]

        # extrair características (features: intensidade do pixel)
        pixels = image_to_feature_vector(image)
        
        # extratir histograma de cores
        hist = extract_color_histogram(image)

        # update the raw images, features, and labels matricies,
        # respectively
        raw_images.append(pixels)
        features.append(hist)
        labels.append(label)

        # show an update every 1,000 images
        if verbose and i > 0 and i % 100 == 0:
            print("[INFO] processed {}/{}".format(i, len(images_path)))
    
    # separa dados de treinos e de teste
    (train_rpi, test_rpi, train_rpi_lbl, test_rpi_lbl) = train_test_split(raw_images, labels, test_size=teste_size, random_state=42)
    (train_his, test_his, train_his_lbl, test_his_lbl) = train_test_split(features, labels, test_size=teste_size, random_state=42)

    # treinar modelo 1
    result_rpi = train_knn(name_model_rpi, num_neighbors, num_jobs, train_rpi, train_rpi_lbl, test_rpi, test_rpi_lbl)
    
    # treinar modelo 2
    result_his = train_knn(name_model_his, num_neighbors, num_jobs, train_his, train_his_lbl, test_his, test_his_lbl)
    
    # gravar classes únicas
    classes = list(set(labels))
    classes.sort()
            
    with open(name_classes, 'w') as f: 
        for c in classes:
            f.write( '{}\n'.format(c))
    
    return classes, result_rpi, result_his
