import datetime
import time
import albumentations as A
from torch.utils.data import Dataset
from dataset import DATASET
from utils.norm_utils import *
from LANet import LANet
from utils.loss import lovasz_softmax
from torch.utils.tensorboard import SummaryWriter

def train_one_epoch(model, optimizer, data_loader, device):
    epoch_loss = 0
    model.train()

    for i, (image, target) in enumerate(data_loader):
        image, target = image.to(device), target.to(device)

        optimizer.zero_grad()
        y_pred = model(image)
        out_loss = lovasz_softmax(y_pred, target)

        out_loss.backward()
        optimizer.step()

        epoch_loss += out_loss.item()

    epoch_loss = epoch_loss / len(data_loader)
    return epoch_loss

def evaluate(model, data_loader, device):
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for i, (image, target) in enumerate(data_loader):
            image, target = image.to(device), target.to(device)

            y_pred = model(image)
            out_loss = lovasz_softmax(y_pred, target)
            epoch_loss += out_loss.item()

            y_pred = torch.argmax(y_pred, dim=1)
            y_pred = y_pred.unsqueeze(0)

            y_true = target.cpu().numpy()
            y_pred = y_pred.cpu().numpy()

            y_pred = y_pred > 0.5
            y_pred = y_pred.reshape(-1)
            y_pred = y_pred.astype(np.uint8)

            y_true = y_true > 0.5
            y_true = y_true.reshape(-1)
            y_true = y_true.astype(np.uint8)

            dice_y = dice_score(y_true, y_pred)
            jac_y = miou_score(y_true, y_pred)

    epoch_loss = epoch_loss / len(data_loader)
    return epoch_loss, dice_y, jac_y


if __name__ == "__main__":

    """ Directories """
    create_dir("files")

    """ Training logfile """
    train_log_path = "train_kvasir_log.txt"
    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        train_log = open("train_kvasir_log.txt", "w")
        train_log.write("\n")
        train_log.close()

    """ Record Date & Time """
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)

    """ Hyperparameters """
    size = (256, 256)
    batch_size = 16
    num_epochs = 300
    lr = 1e-4
    # segmentation nun_classes + background
    num_classes = 1
    checkpoint_path = "checkpoint_kvasir.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ Dataset """
    path = "dataset_PATH"
    (train_x, train_y), (valid_x, valid_y) = load_dataset(path)
    train_x, train_y = shuffling(train_x, train_y)

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter('logs_path')

    train_dataset = DATASET(train_x, train_y, size, transform='Your transform method')
    valid_dataset = DATASET(valid_x, valid_y, size, transform=None)

    num_workers = 0
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(valid_dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             shuffle=False)

    # create model
    model = Ablation().to(device)
    model.load_encoder_weight()
    for param in model.encoder.parameters():
        param.requires_grad = False

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_name = "Lovas softmax"

    data_str = f"Hyperparameters:\nImage Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    best_valid_loss = float('inf')
    start_time = time.time()
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        valid_loss, dice_y, jac_y = evaluate(model, val_loader, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            data_str = f"Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print_and_save(train_log_path, data_str)