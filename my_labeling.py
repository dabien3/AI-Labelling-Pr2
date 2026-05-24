__authors__ = 'TU_NOMBRE'
__group__ = 'TU_GRUPO'

import numpy as np
from KNN import KNN
from Kmeans import KMeans, get_colors
from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval
import time

def testKNN(train_imgs, train_class_labels, test_imgs, test_class_labels):

    dist_list = ['euclidean', 'cityblock', 'cosine']


    base_options = {
        "knn_size_data": (40, 40),
        "f_space": "mean"
    }

    knn_base = KNN(train_imgs, train_class_labels, base_options)

    X_train_base = knn_base.train_data


    for k in range(3, 8):

        for d in dist_list:

            for q in range(10, 41): 

                options = {
                    "knn_size_data": (40, 40),
                    "f_space": "mean",
                    "quadrants": q,
                    "dist": d
                }

                knn_model = KNN(train_imgs, train_class_labels, options)

                print("Dist ->", d,
                      "Quadrants ->", q,
                      "k ->", k + 1)

                Get_shape_accuracy(
                    knn_model,
                    test_imgs,
                    test_class_labels,
                    k + 1
                )
        

def Get_shape_accuracy(knn, test_imgs, test_class_labels, k):
    
    # la IA intenta adivinar qué es cada prenda
    start = time.time()
    predicciones = knn.predict(test_imgs, k)
    end = time.time()
    # contamos aciertos
    aciertos = 0
    total = len(test_class_labels)
    
    for i in range(total):
        if predicciones[i] == test_class_labels[i]:
            aciertos = aciertos + 1
    
    nota = (aciertos / total) * 100
    
    print("K =", k, ": Eficacia:", nota, "%", "Time: ", end-start, " s")

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
    lista_stat = []
    lista_iter = []
    lista_time = []

    for k in range(2, k_max + 1):

        stat_total = 0
        iter_total = 0
        time_total = 0

        for img in images:
            #para cada foto ejecutamos Kmeans, guardamos el WCD, tiempo y iteracion de cada uno
            mi_km = KMeans(img, K=k, options=options)

            start = time.time()
            mi_km.fit()
            end = time.time()
            if options["stat"] == "WCD":
                stat_total += mi_km.withinClassDistance()
            elif options["stat"] == "ICD":
                stat_total += mi_km.interClassDistanceCentroids()
            elif options["stat"] == "Fisher":
                stat_total += mi_km.fisherDiscriminant()
            iter_total += mi_km.num_iter
            time_total += (end - start)

        lista_stat.append(stat_total / len(images))
        lista_iter.append(iter_total / len(images))
        lista_time.append(time_total / len(images))

    return lista_stat, lista_iter, lista_time

if __name__ == '__main__':

    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    options = {
        "knn_size_data": (40,40),
        "f_space": "mean",
        "quadrants": 20,
        "dist" : "euclidean"
    }

    '''
    
    testKNN(train_imgs, train_class_labels, test_imgs, test_class_labels)

    '''

    print("\n[TEST 1] Get_shape_accuracy, Eficacia (KNN):")
    knn_model = KNN(train_imgs, train_class_labels, options)
    Get_shape_accuracy(knn_model, test_imgs, test_class_labels, 4)

    

    '''

    print("\n[TEST 2] Get_color_accuracy, Precisión de Color (K-Means):")
    options={'km_init': 'first'}
    acc_color = Get_color_accuracy(cropped_images, color_labels, options=options, K = 3)
    print(f"ÍNDICE DE CALIDAD: {acc_color}%")

    '''
    
    '''
    
    print("\n[TEST 3] Kmeans_statistics (WCD):")
    options={
        'km_init': 'first',
        "stat": "ICD"
        }
    stat, iters, t = Kmeans_statistics(8, cropped_images, options=options)
    for i in range(len(stat)):
        print(f"K={i+2} | {options["stat"]}={stat[i]:.2f} | Time={t[i]:.4f}s | Iter={iters[i]:.1f}")
        
    '''