__authors__ = ["1707361, 1709928, 1711116"]
__group__ = '87'

import numpy as np
from KNN import KNN
from Kmeans import KMeans, get_colors
from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval
import time

def retrieval_by_shape(images, shape_labels, query):
    matched_images= [] # list where we save the images
    for i, labels in enumerate(shape_labels): # we iterate the indices and the labels
        if query in labels:
            matched_images.append(images[i])
        
    return matched_images

def retrieval_by_color(images, color_labels, query):
    matched_indices = []
    for i, labels in enumerate(color_labels):
        if all(color in labels for color in query): # i didnt know how to find all the colors in the label of an image, but then I find all() that comproves if all the colors in the query are in the labels of the image (return True if all are found). With just one color in the query it won't be necessary use all 
            matched_indices.append(i)
            
    for i in matched_indices: # then like before we save the images that match the query
        matched_images = images[i] 
        
    return matched_images

def retrieval_combined(images, shape_labels, color_labels, query_shape, query_colors): 

    matched_images_by_shape = retrieval_by_shape(images, shape_labels, query_shape)
    matched_images_by_color = retrieval_by_color(images, color_labels, query_colors)
    
    matched_images = []
    # we want to find the images that match both the shape and the color, so we iterate the images that match the shape and we check if they are in the list of images that match the color, if they are we save them in the list of matched_images
    for img in matched_images_by_shape:
        if img in matched_images_by_color:
            matched_images.append(img)
    
    return matched_images
    
def Get_shape_accuracy(knn, test_imgs, test_class_labels, k):
    
    # la IA intenta adivinar qué es cada prenda
    predicciones = knn.predict(test_imgs, k)
    
    # contamos aciertos
    aciertos = 0
    total = len(test_class_labels)
    
    for i in range(total):
        if predicciones[i] == test_class_labels[i]:
            aciertos = aciertos + 1
    
    nota = (aciertos / total) * 100
    
    print("K =", k, ": Eficacia:", nota, "%")

def Get_color_accuracy(lista_recortes, color_labels, options, K):
    total = len(lista_recortes)
    score = 0

    for i in range(total):
        foto = lista_recortes[i]
        # para cada foto ejecutamos Kmeans
        ia_color = KMeans(foto, K=K, options=options)
        ia_color.fit()
        #sea A el resultado en colores para esa foto del Kmeans, y B el GT
        A = set(get_colors(ia_color.centroids))
        B = set(color_labels[i])

        #nos da cuantos colores ha acertado Kmeans de los que estan en el GT
        score += len(A & B) / len(A)
        
    return 100 * score / total

def Kmeans_statistics(k_max, images, options):
    #listas para guardar resultados Kmeans
    lista_wcd = []
    lista_iter = []
    lista_time = []

    for k in range(2, k_max + 1):

        wcd_total = 0
        iter_total = 0
        time_total = 0

        for img in images:
            #para cada foto ejecutamos Kmeans, guardamos el WCD, tiempo y iteracion de cada uno
            mi_km = KMeans(img, K=k, options=options)

            start = time.time()
            mi_km.fit()
            end = time.time()
            
            wcd_total += mi_km.withinClassDistance()
            iter_total += mi_km.num_iter
            time_total += (end - start)

        lista_wcd.append(wcd_total / len(images))
        lista_iter.append(iter_total / len(images))
        lista_time.append(time_total / len(images))

    return lista_wcd, lista_iter, lista_time

if __name__ == '__main__':

    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    print("\n[TEST 1] Get_shape_accuracy, Eficacia (KNN):")
    knn_model = KNN(train_imgs, train_class_labels)
    Get_shape_accuracy(knn_model, test_imgs, test_class_labels, 4)

    print("\n[TEST 2] Get_color_accuracy, Precisión de Color (K-Means):")
    acc_color = Get_color_accuracy(cropped_images, color_labels, options={'km_init': 'custom2'}, K = 4)
    print(f"ÍNDICE DE CALIDAD: {acc_color}%")

    print("\n[TEST 3] Kmeans_statistics (WCD):")
    wcd, iters, t = Kmeans_statistics(8, cropped_images, options={'km_init': 'custom2'})
    for i in range(len(wcd)):
        print(f"K={i+2} | WCD={wcd[i]:.2f} | Time={t[i]:.4f}s | Iter={iters[i]:.1f}")
    
    print("\n[TEST 4] Retrieval_by_shape (Búsqueda por Forma):")
    # 1. KNN predicts the shape labels for the test images
    prediction_knn = knn_model.predict(test_imgs, k=4)
    
    # 2. We query for "Shirts"
    query_form = "Shirts"
    results_form = retrieval_by_shape(test_imgs, prediction_knn, query_form)
    
    print(f"There're {len(results_form)} with shape {query_form}")