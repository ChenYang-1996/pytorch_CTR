#用户行为序列建模：DIN, DIEN, CAN
import torch
import tqdm
import numpy as np
import torch.utils.data as Data
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
writer = SummaryWriter('../logs')
from sklearn.model_selection import train_test_split
from torchfm.model.din import DeepInterestNet
from torchfm.model.dien import DeepInterestEvolutionNet
from torchfm.dataset.AmazonBook import AmazonBookPreprocess

data = pd.read_csv('../torchfm/dataset/AmazonBook/amazon-books-100k.txt')

def get_dataset(name, path):
    if name == 'amazon-books-100k':
        return AmazonBookPreprocess(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.max().max() #被点击次数最多的cate

    if name == 'din': ##########################debuging
        print("Model:DIN")
        return DeepInterestNet(
        feature_dim=field_dims, embed_dim=8, mlp_dims=[64, 32], dropout=0.2)
    elif name == 'dien': ##########################debuging
        print("Model:DIEN")
        return DeepInterestEvolutionNet(
        feature_dim=field_dims, embed_dim=4, hidden_size=4, mlp_dims=(64, 32), dropout=0.2)
    else:
        raise ValueError('unknown model name: ' + name)


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0
    return total_loss #yc

def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)

def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dataset = get_dataset(dataset_name, dataset_path)
    data_x = dataset.iloc[:, :-1]
    data_y = dataset.label.values
    tmp_x, test_x, tmp_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=42, stratify=data_y)
    train_x, val_x, train_y, val_y = train_test_split(tmp_x, tmp_y, test_size=0.25, random_state=42, stratify=tmp_y)
    train_x = torch.from_numpy(train_x.values).long()
    val_x = torch.from_numpy(val_x.values).long()
    test_x = torch.from_numpy(test_x.values).long()

    train_y = torch.from_numpy(train_y).long()
    val_y = torch.from_numpy(val_y).long()
    test_y = torch.from_numpy(test_y).long()

    train_set = Data.TensorDataset(train_x, train_y)
    val_set = Data.TensorDataset(val_x, val_y)
    test_set = Data.TensorDataset(test_x, test_y)

    train_data_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=6, shuffle=True) #num_workers=8
    valid_data_loader = DataLoader(dataset=val_set, batch_size=batch_size, num_workers=6, shuffle=False) #num_workers=8
    test_data_loader = DataLoader(dataset=test_set, batch_size=batch_size, num_workers=6, shuffle=False) #num_workers=8
    model = get_model(model_name, dataset).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=2, save_path=f'{save_dir}/{model_name}_{dataset_name}.pt') #2
    for epoch_i in range(epoch):
        #train(model, optimizer, train_data_loader, criterion, device)
        loss = train(model, optimizer, train_data_loader, criterion, device) #yc
        auc = test(model, valid_data_loader, device)
        print('epoch:', epoch_i, 'training loss:', loss)
        print('epoch:', epoch_i, 'validation: auc:', auc)
        writer.add_scalar('loss', scalar_value=int(loss), global_step=epoch_i)
        writer.add_scalar('acc', scalar_value=auc, global_step=epoch_i)
        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break
    auc = test(model, test_data_loader, device)
    print(f'test auc: {auc}')



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='amazon-books-100k')
    # parser.add_argument('--dataset_path', help='criteo/train.txt, avazu/train, or ml-1m/ratings.dat')
    parser.add_argument('--dataset_path', default=data)
    parser.add_argument('--model_name', default='dien') ##########
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128) #2048
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='../chkpt')
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir)
