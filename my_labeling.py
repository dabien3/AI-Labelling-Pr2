__authors__ = ["1707361, 1709928, 1711116"]
__group__ = '87'

import numpy as np
from KNN import KNN
from Kmeans import KMeans, get_colors
from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval
import time
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
    predicciones, _ = knn.predict(test_imgs, k)
    
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

    # print("\n[TEST 1] Get_shape_accuracy, Eficacia (KNN):")
    knn_model = KNN(train_imgs, train_class_labels)
    # Get_shape_accuracy(knn_model, test_imgs, test_class_labels, 4)

    # print("\n[TEST 2] Get_color_accuracy, Precisión de Color (K-Means):")
    # acc_color = Get_color_accuracy(cropped_images, color_labels, options={'km_init': 'custom2'}, K = 4)
    # print(f"ÍNDICE DE CALIDAD: {acc_color}%")

    # print("\n[TEST 3] Kmeans_statistics (WCD):")
    # wcd, iters, t = Kmeans_statistics(8, cropped_images, options={'km_init': 'custom2'})
    # for i in range(len(wcd)):
    #     print(f"K={i+2} | WCD={wcd[i]:.2f} | Time={t[i]:.4f}s | Iter={iters[i]:.1f}")
    
    '''We start here with the retrieval tests, we will test the retrieval by shape, color and combined'''

    #Obtaining predictions for test images:
    print("Calculating KNN predictions for test images")
    form_predictions, form_trust = knn_model.predict(test_imgs, k=4) #in the 2nd point we see the optim K for KNN it's 4
    color_predictions = []
    color_probs = []
    print("Calculating Kmeans predictions for test images")
    # for each test image we do Kmeans and we get the colors and the confidence score for each color, 
    # the confidence score is the proportion of pixels of that color in the image, as we said before
    # we store the color predictions and the confidence scores in color_predictions and color_probs, that will be used in the retrieval by color and combined
    for i in range(len(test_imgs)):
        img = test_imgs[i]
        # the ideal K for Kmeans is 4, as we have seen in the quality assessment, but if we didn't know what k its optimal we could 
        # test with different K and see which one gives us better results in the retrieval tests, we could find it with findbestK, 
        # in this case we won't use it cause it's a test to see how good our Kmeans is with the optimal K, but in a real scenario we 
        # would have to find it first and it's so slow. Then 'custom2' is our custom initialization method for the centroids.
        # 
        
        km = KMeans(img, K=4, options={'km_init': 'custom2'}) 
        km.fit()   
        # get_colors transforms the 4 centroids (RGB values) into color names
        colors = get_colors(km.centroids)
        color_predictions.append(colors)
        # Now we calculate what proportion of the image each color occupies. 'km.labels' is an array where each pixel has the index of its assigned cluster.
        total_pixels = len(km.labels)
        dic = {}
        for j, color_name in enumerate(colors):
            # Count how many pixels belong to cluster j, then divide by total pixels to get the proportion
            proportion = float(np.sum(km.labels == j) / total_pixels)

            # We use get() with a default of 0.0 because two clusters can map to the same color name. Then we add the same proportions of the color together instead of overwriting the first one.
            dic[color_name] = dic.get(color_name, 0.0) + proportion
        color_probs.append(dic)

    # Retrieval by Shape:
    query_form = ["Dresses", "Shirts", "Socks"]
    print(f"\n[TEST 1.1] Searching '{query_form}'")
    res_shape = res_shape = retrieval_by_shape(test_imgs, form_predictions, query_form, shape_probs=form_trust)
    print(f"-> Encontradas {len(res_shape)} imágenes.")
    
    if len(res_shape) > 0:
        visualize_retrieval(res_shape, 15, title=f"Retrieval Shape: {query_form}")

    # Retrieval by Color:
    query_color = ["Red", "Black"]
    print(f"\n[TEST 1.2] Searching color '{query_color}'")
    res_color = retrieval_by_color(test_imgs, color_predictions, query_color, color_probs=color_probs)
    print(f"-> Encontradas {len(res_color)} imágenes.")
    
    if len(res_color) > 0:
        visualize_retrieval(res_color, 15, title=f"Retrieval Color: {query_color}")

    #  Retrieval Combined:
    query_f_comb = ["Dresses", "Shirts"]
    query_c_comb = ["White", "Red"]
    print(f"\n[TEST 1.3] Searching '{query_c_comb}' of color '{query_f_comb}'")
    res_combined = retrieval_combined(test_imgs, form_predictions, color_predictions, query_shape = query_f_comb, query_colors=query_c_comb, shape_probs = form_trust, color_probs=color_probs)
    
    if len(res_combined) > 0:
        visualize_retrieval(res_combined, 15, title="Combinada: " + f"{query_f_comb} + {query_c_comb}")