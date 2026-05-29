__authors__ = ["1707361, 1709928, 1711116"]
__group__ = '87'

import numpy as np
from KNN import KNN
from Kmeans import KMeans, get_colors
from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval, visualize_k_means
import time

def testKNN(train_imgs, train_class_labels, test_imgs, test_class_labels): #funcion propia no se entrega

    dist_list = ['euclidean', 'cityblock']


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
        

import utils

def retrieval_by_shape(images, shape_labels, query_shape, shape_probs=None):
    """
    Returns images that match with the variable query_shape, that can be sorted by confidence.
    shape_probs: optional confidence score per image from KNN. If provided, results are sorted from most to least confident.
    """
    matched_images = []
    matched_scores = []

    for i in range(len(images)):
        # We choose only the images that match the query shape
        if shape_labels[i] in query_shape:
            matched_images.append(images[i])
            # If shape_probs is provided, we use it as the confidence score for sorting
            if shape_probs is not None:
                matched_scores.append(float(shape_probs[i]))
            else:
                matched_scores.append(1.0)
    #As we end the selection of the images, if we have confidence scores we sort the results from most to least confident
    if shape_probs is not None:
         #we sort the images with a bubble sort, this same algorithm will be used in the other retrieval functions
         for i in range(len(matched_scores)):
            for j in range(i + 1, len(matched_scores)):
                if matched_scores[j] > matched_scores[i]:
                    temp_score = matched_scores[i]
                    matched_scores[i] = matched_scores[j]
                    matched_scores[j] = temp_score
                    temp_img = matched_images[i]
                    matched_images[i] = matched_images[j]
                    matched_images[j] = temp_img

    return matched_images

def retrieval_by_color(images, color_labels, query, color_probs=None):
    """
    Returns images that contain ALL the queried colors, optionally sorted by how many presence they have.
    - color_probs: optional dict per image mapping color -> how many pixels of that color in proportion 
    (for example: {"Red": 0.45, "Black": 0.30}). 
    If provided, results will be sorted from most to least presence of the requeried colors.
    """
    matched_images = []
    matched_scores = []
    for i in range(len(images)):
         # We comprove iof all the colors in the query are present in the image, all() it's a python function that returns true if all the elements of the iterable are true, in this case we check if all the colors in the query are in the color_labels of the image
        if all(color in color_labels[i] for color in query):
            matched_images.append(images[i])
            
            if color_probs is not None:
                # Score = sum of pixel proportions for each color
                score = 0.0
                for color in query:
                    if color in color_probs[i]:
                        score += color_probs[i][color]
                matched_scores.append(score)
            else:
                matched_scores.append(1.0)
    #we sort with  bubble sort
    if color_probs is not None:
         for i in range(len(matched_scores)):
            for j in range(i + 1, len(matched_scores)):
                if matched_scores[j] > matched_scores[i]:
                    temp_score = matched_scores[i]
                    matched_scores[i] = matched_scores[j]
                    matched_scores[j] = temp_score
                    temp_img = matched_images[i]
                    matched_images[i] = matched_images[j]
                    matched_images[j] = temp_img

    return matched_images

def retrieval_combined(images, shape_labels, color_labels, query_shape, query_colors, shape_probs=None, color_probs=None):
    '''
    Returns images that matach both queries, optionally sorted by a combined confidence score of shape and color.
    As it saids the name, it's a combination of the previous retrievals, in this function the especial it's that we gave a weight
    to the shape and color confidence scores, as we will see in the tests, KNN is more accurate than Kmeans, so we give more weight to 
    the shape score than to the color score, but this is something that could be tested with different weights and see which one gives 
    better results in the retrieval tests
    '''
    matched_images = []
    matched_scores = []


    for i in range(len(images)):
        form_match  = shape_labels[i] in query_shape
        color_match = all(color in color_labels[i] for color in query_colors)

        if form_match and color_match:
            matched_images.append(images[i])
            if shape_probs is not None:
                score_form  = float(shape_probs[i])
            else:
                score_form  = 1.0
            if color_probs is not None:
                score_color = 0.0
                for color in query_colors:
                    if color in color_probs[i]:
                        score_color = score_color + color_probs[i][color]
            else:
                score_color = 1.0
            matched_scores.append(0.65 * score_form + 0.35 * score_color) #as we will see, KNN is more accurate than Kmeans, so we give more weight to the shape score than to the color score, but this is something that could be tested with different weights and see which one gives better results in the retrieval tests
    #We sort the results with bubble sort, from most to least confident
    if matched_scores:
         for i in range(len(matched_scores)):
            for j in range(i + 1, len(matched_scores)):
                if matched_scores[j] > matched_scores[i]:
                    temp_score = matched_scores[i]
                    matched_scores[i] = matched_scores[j]
                    matched_scores[j] = temp_score
                    temp_img = matched_images[i]
                    matched_images[i] = matched_images[j]
                    matched_images[j] = temp_img

    return matched_images
    
def Get_shape_accuracy(knn, test_imgs, test_class_labels, k):
    
    # la IA intenta adivinar qué es cada prenda
    start = time.time()
    predicciones = knn.predict(test_imgs, k)
    end = time.time()
    predicciones, _ = knn.predict(test_imgs, k)
    
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

    '''

    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    options = {
        "knn_size_data": (40,40),
        "f_space": "mean",
        "quadrants": 40,
        "dist" : "euclidean"
    }


    km = KMeans(cropped_images[3], 4, {'km_init': 'custom2'})
    km.fit()
    print(color_labels[3])
    print(get_colors(km.centroids))
    visualize_k_means(km, cropped_images[3].shape)

    
    testKNN(train_imgs, train_class_labels, test_imgs, test_class_labels)

    print("\n[TEST 1] Get_shape_accuracy, Eficacia (KNN):")
    knn_model = KNN(train_imgs, train_class_labels, options)
    plot_pca(knn_model, train_class_labels)
    Get_shape_accuracy(knn_model, test_imgs, test_class_labels, 4)

    print("\n[TEST 2] Get_color_accuracy, Precisión de Color (K-Means):")
    options={'km_init': 'first'}
    acc_color = Get_color_accuracy(cropped_images, color_labels, options=options, K = 4)
    print(f"ÍNDICE DE CALIDAD, first: {acc_color}%")

    acc_color = Get_color_accuracy(cropped_images, color_labels, options={'km_init': 'random'}, K = 4)
    print(f"ÍNDICE DE CALIDAD, random: {acc_color}%")
    acc_color = Get_color_accuracy(cropped_images, color_labels, options={'km_init': 'custom'}, K = 4)
    print(f"ÍNDICE DE CALIDAD, equally: {acc_color}%")
    acc_color = Get_color_accuracy(cropped_images, color_labels, options={'km_init': 'custom2'}, K = 4)
    print(f"ÍNDICE DE CALIDAD, kmeans++: {acc_color}%")
    
    print(f"ÍNDICE DE CALIDAD: {acc_color}%")

    print("\n[TEST 3] Kmeans_statistics (WCD):")
    options={
        'km_init': 'first',
        "stat": "WCD"
        }
    stat, iters, t = Kmeans_statistics(8, imgs, options=options)
    print("First: ")
    for i in range(len(stat)):
        print(f"K={i+2} | {options["stat"]}={stat[i]:.2f} | Time={t[i]:.4f}s | Iter={iters[i]:.1f}")
        options={
        'km_init': 'random',
        "stat": "WCD"
        }
    stat, iters, t = Kmeans_statistics(8, imgs, options=options)
    print("Random: ")
    for i in range(len(stat)):
        print(f"K={i+2} | {options["stat"]}={stat[i]:.2f} | Time={t[i]:.4f}s | Iter={iters[i]:.1f}")
        options={
        'km_init': 'custom',
        "stat": "WCD"
        }
    stat, iters, t = Kmeans_statistics(8, imgs, options=options)
    print("Equally: ")
    for i in range(len(stat)):
        print(f"K={i+2} | {options["stat"]}={stat[i]:.2f} | Time={t[i]:.4f}s | Iter={iters[i]:.1f}")
        options={
        'km_init': 'custom2',
        "stat": "WCD"
        }
    stat, iters, t = Kmeans_statistics(8, imgs, options=options)
    print("Kmeans++: ")
    for i in range(len(stat)):
        print(f"K={i+2} | {options["stat"]}={stat[i]:.2f} | Time={t[i]:.4f}s | Iter={iters[i]:.1f}")
    '''
    
        
