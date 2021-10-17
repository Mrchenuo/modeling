import argparse
from lib2to3.pgen2.grammar import op

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.simple_model as simple_model
from parse_config import ConfigParser
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def main(config):
    logger = config.get_logger('test')
    # build model architecture
    net = config.init_obj('arch', simple_model)
    logger.info(net)

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['test_data_path'],
        batch_size=64,
        shuffle=False,
        validation_split=0.0,
        phase='test',
        num_workers=2
    )

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(net)
    net.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    net.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    output_all = None
    target_all = None
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device, dtype=torch.float), target.to(device)
            output = net(data)

            if i == 0:
                output_all = output
                target_all = target
            else:
                output_all = torch.cat([output_all, output], dim=0)
                target_all = torch.cat([target_all, target], dim=0)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size
    # print(output_all)
    # print(target_all)
    # dif = np.absolute(output_all - target_all)
    # print(dif)

    # list转dataframe
    # df = pd.DataFrame(output_all.numpy(), columns=['pred_pIC50'])
    # # 保存到本地excel
    # df.to_excel("test_pIC50.xlsx", index=False)
    #
    # df = pd.DataFrame(target_all.numpy(), columns=['real_pIC50'])
    # # 保存到本地excel
    # df.to_excel("target_pIC50.xlsx", index=False)
    # bg = op.load_workbook(r"D:/GitHub/modeling/D/data/ERα_activity.xlsx")  # 应先将excel文件放入到工作目录下
    # sheet = bg["test"]
    # for i in range(1, len(num_list) + 1):
    #     sheet.cell(i, 1, num_list[i - 1])

    # 画预测值与label的散点图
    output_list = output_all.numpy().tolist()
    target_list = target_all.numpy().tolist()
    out = [b for a in output_list for b in a]
    tar = [b for a in target_list for b in a]

    # idx = [i + 1 for i in range(len(out))]
    # for i in range(len(target_list)):
    #     idx.extend(target_list[i])
    # plt.figure(figsize=(10, 10), dpi=100)
    # plt.plot(idx, out, linestyle='-', marker='o', c='g')
    # plt.scatter(idx, out, marker='o', c='g')
    # plt.plot(idx, tar, linestyle='--', marker='.', c='b')
    # plt.scatter(idx, tar, marker='.', c='b')

    out_np = np.array(out)
    tar_np = np.array(tar)
    dif = np.abs(tar_np - out_np)
    idx_need = np.where(dif < 0.2)
    out_select = out_np[idx_need]
    tar_select = tar_np[idx_need]
    print(len(idx_need))

    idx = [i + 1 for i in range(len(out_select))]
    # for i in range(len(target_list)):
    #     idx.extend(target_list[i])
    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(idx, out_select, label='预测值', linestyle='-', marker='o', c='g')
    plt.scatter(idx, out_select, marker='o', c='g')
    plt.plot(idx, tar_select, label='真实值', linestyle='--', marker='.', c='b')
    plt.scatter(idx, tar_select, marker='.', c='b')
    plt.ylabel('生物活性值', fontdict={'size': 16})
    plt.xlabel('预测样本编号', fontdict={'size': 16})
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.legend(loc="lower right", fontsize=16)
    plt.savefig("pred.png")
    plt.show()

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='detection')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=r"saved/models/Detection/1016_204223\model_best.pth",
                      type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
