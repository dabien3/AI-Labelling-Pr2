from KNN import __authors__, __group__, KNN
from utils_data import *

if __name__ == "__main__":
    train_imgs_grayscale, train_class_labels, _, test_imgs_grayscale, test_class_labels, _ = read_dataset(
        root_folder = "./images/" , gt_json="./images/gt.json", with_color = False
    )
    knn = KNN(train_imgs_grayscale, train_class_labels)
    knn.get_k_neighbours(np.array([test_imgs_grayscale[4]]), 7)
    print(knn.neighbors)

    test_imgs_grayscale = read_one_img("./images/test/1670.jpg", 60, 80, False)
    knn.get_k_neighbours(np.array([test_imgs_grayscale]), 7)
    print(knn.neighbors)