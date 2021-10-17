import os.path
import pickle
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm
from xgboost import XGBClassifier
from simple_model import Net2
from data_loader.data_loaders import RIDataLoader
import torch.nn.functional as F

# col_name = ['Caco-2']
col_name = ['Caco-2', 'CYP3A4', 'hERG', 'HOB', 'MN']

model_1 = pickle.load(open(os.path.join(r'D:\GitHub\modeling\D\ans3', "Caco-2.pickle"), "rb"))
model_2 = pickle.load(open(os.path.join(r'D:\GitHub\modeling\D\ans3', "CYP3A4.pickle"), "rb"))
model_3 = pickle.load(open(os.path.join(r'D:\GitHub\modeling\D\ans3', "hERG.pickle"), "rb"))
model_4 = pickle.load(open(os.path.join(r'D:\GitHub\modeling\D\ans3', "HOB.pickle"), "rb"))
model_5 = pickle.load(open(os.path.join(r'D:\GitHub\modeling\D\ans3', "MN.pickle"), "rb"))


# loaded_model = list()
# for col in col_name:
#     ld_mod = pickle.load(open(os.path.join(r'D:\GitHub\modeling\D\ans3', col + ".pickle"), "rb"))
#     loaded_model.extend(ld_mod)


def ans2_pred(test_X):
    data_loader = RIDataLoader(
        test_X,
        batch_size=64,
        shuffle=False,
        validation_split=0.0,
        num_workers=0
    )
    net = Net2()
    checkpoint = torch.load(r'D:\GitHub\modeling\D\ans2\data_mining\saved\models\Detection\1016_204223\model_best.pth')
    state_dict = checkpoint['state_dict']
    net.load_state_dict(state_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    net.eval()
    output_list = list()
    target_list = list()
    total_loss = 0.0
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device, dtype=torch.float), target.to(device)
            output = net(data)
            output = output.squeeze()
            output_list.append(output.numpy())
            target = target.squeeze()
            target_list.append(target.numpy())

            loss = F.mse_loss(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
        n_samples = len(data_loader.sampler)
        # print('loss: ', total_loss / n_samples)
    return output_list


def ans3_pred(test_X):
    # 标准化
    ss = StandardScaler()
    test_X = ss.fit_transform(test_X)

    pred_list = list()
    # test_X = test_X_df.iloc[:, 1:].values  # 去掉第一列
    pred_1 = model_1.predict(test_X)
    pred_list.append(pred_1)
    # tmp_col_name = 'Caco-2'
    # df = pd.DataFrame(pred_1, columns=[tmp_col_name])
    # # 保存到本地excel
    # df.to_excel(tmp_col_name + ".xlsx", index=False)

    pred_2 = model_2.predict(test_X)
    pred_list.append(pred_2)
    # tmp_col_name = 'CYP3A4-2'
    # df = pd.DataFrame(pred_1, columns=[tmp_col_name])
    # # 保存到本地excel
    # df.to_excel(tmp_col_name + ".xlsx", index=False)

    pred_3 = model_3.predict(test_X)
    pred_list.append(pred_3)
    # tmp_col_name = 'hERG'
    # df = pd.DataFrame(pred_1, columns=[tmp_col_name])
    # # 保存到本地excel
    # df.to_excel(tmp_col_name + ".xlsx", index=False)

    pred_4 = model_4.predict(test_X)
    pred_list.append(pred_4)
    # tmp_col_name = 'HOB'
    # df = pd.DataFrame(pred_1, columns=[tmp_col_name])
    # # 保存到本地excel
    # df.to_excel(tmp_col_name + ".xlsx", index=False)

    pred_5 = model_5.predict(test_X)
    pred_list.append(pred_5)
    # tmp_col_name = 'MN'
    # df = pd.DataFrame(pred_1, columns=[tmp_col_name])
    # # 保存到本地excel
    # df.to_excel(tmp_col_name + ".xlsx", index=False)

    # for i in len(loaded_model):
    #     pred = loaded_model.predict(test_X)
    #     pred_list.extend(pred)
    # # load model from file
    # loaded_model = pickle.load(open(col_name[idx - 1] + ".pickle", "rb"))
    # # 测试结果
    # test_X_df = pd.read_excel(r"D:\GitHub\modeling\D\ans3\Molecular_Descriptor.xlsx", engine='openpyxl',
    #                           sheet_name='test')
    return pred_list


if __name__ == '__main__':
    test_X_df = pd.read_excel(r"D:\GitHub\modeling\D\ans3\Molecular_Descriptor.xlsx", engine='openpyxl',
                              sheet_name='training')
    # test_X_df = test_X_df[idx_one]
    test_X = test_X_df.iloc[:, 1:].values  # 去掉第一列
    pred_list = ans3_pred(test_X)
