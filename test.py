from KNN import __authors__, __group__, KNN
from utils import *
from utils_data import *


train_imgs = read_dataset(
    root_folder='./images/',
    gt_json='./images/gt.json',
    with_color=False
)

for ix, (train_imgs, train_labels) in enumerate(train_imgs['input']):
            knn = KNN(train_imgs, train_labels)
            knn.get_k_neighbours(train_imgs['test_input'][ix][0], train_imgs['rnd_K'][ix])