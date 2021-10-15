import argparse
import torch
from torch import nn
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.simple_model as simple_model
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')
    # build model architecture
    net = config.init_obj('arch', simple_model)
    logger.info(net)

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['test_data_dir'],
        batch_size=64,
        image_size=None,
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
    args.add_argument('-r', '--resume', default=r"saved_model\model_best.pth",
                      type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
