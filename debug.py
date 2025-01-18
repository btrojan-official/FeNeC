import torch

from model import Knn_Kmeans_Logits 
from utils.other import load_data
from configs.config import config

device =  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

model = Knn_Kmeans_Logits(config, device=device)

for i in range(6):

    X_train, y_train, X_test, y_test, covariances = load_data(task_num=i, dataset_name="resnet", load_covariances=True, load_prototypes=False)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    acc = (torch.sum((y_test.flatten().to(device)==predictions).int()) / X_test.shape[0] * 100).item()

    print(f"Accuracy: {acc} MY\n")
