import os
import time
from operator import add
import cv2
import numpy as np
import torch
from tqdm import tqdm
from LANet import LANet
from utils.norm_utils import create_dir

def load_datasets(path):
    def load_names(path, file_path):
        f = open(file_path, "r")
        file_names = f.read().split("\n")[:-1]
        images = [os.path.join(path, "images", name) for name in file_names]
        masks = [os.path.join(path, "masks", name) for name in file_names]
        return images, masks, file_names

    train_names_path = f"{path}/test.txt"
    valid_names_path = f"{path}/test.txt"

    train_x, train_y, train_names = load_names(path, train_names_path)
    test_x, test_y, test_names = load_names(path, valid_names_path)

    return (train_x, train_y, train_names), (test_x, test_y, test_names)


class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """

    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])


if __name__ == "__main__":
    """ Seeding """
    seeding(42)
    data_name = 'CVC_clinicDB'
    """ set parameters and records"""
    path = f"./datasets/{data_name}"

    checkpoint_path = f"woAFF/{data_name}/checkpoint_clinicDB.pth"
    file = open(f"woAFF/{data_name}/test_clinicDB_results.csv", "w")

    file.write("Jaccard,F1,Recall,Precision,Specificity,Accuracy,F2,Mean Time,Mean FPS\n")

    save_imgs = True
    if save_imgs:
        save_path = f'masks/{data_name}/preds/'
        save_gt = f'masks/{data_name}/gts/'

    """ load datasets """
    (train_x, train_y, train_names), (test_x, test_y, test_names) = load_datasets(path)
    """ Hyperparameters """
    size = (256, 256)

    """ save files """
    create_dir("results")

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    model = LANet()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = CustomDataParallel(model).to(device)
    model.eval()

    # warm-up for cal exact time
    model(torch.randn(4, 3, 256, 256))

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []
    total_score = []
    for i, (x, y,test_name) in tqdm(enumerate(zip(test_x, test_y, test_names)), total=len(test_x)):
        ## Image
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        # img_x = image
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)

        ## GT Mask
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        # save gt masks
        if save_imgs:
            cv2.imwrite(save_gt + test_name, mask)

        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(device)

        with torch.no_grad():
            """ FPS Calculation """
            start_time = time.time()
            pred_y = model(image)

            end_time = time.time() - start_time
            time_taken.append(end_time)

            pred_y = torch.argmax(pred_y, dim=1)

            # pred_y = torch.sigmoid(pred_y)
            pred_y = pred_y.unsqueeze(0)

            score = calculate_metrics(mask, pred_y)
            total_score.append(score)
            metrics_score = list(map(add, metrics_score, score))
            # 保存pred masks
            if save_imgs:
                pred_y = pred_y.cpu().detach().numpy().squeeze(0).squeeze(0)
                pred_y = (pred_y * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(save_path + test_name, pred_y)