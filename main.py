from argparse import ArgumentParser
from torch import device
from test import test
from train import train


if __name__ == '__main__':
    argumentParser = ArgumentParser()

    argumentParser.add_argument('--mode', type=str, choices=('train', 'test'), default='train')

    argumentParser.add_argument('--plus', type=int, choices=(0, 1), default=1)

    argumentParser.add_argument('--device', type=int, default=0)

    argumentParser.add_argument('--batch_size', type=int, default=4)
    argumentParser.add_argument('--num_workers', type=int, default=15)
    argumentParser.add_argument('--image_size', nargs='+', type=int, default=(256, 256))

    argumentParser.add_argument('--iaff_num_epochs', type=int, default=200)
    argumentParser.add_argument('--iaff_lr', type=float, default=1e-5)
    argumentParser.add_argument('--iaff_weight_decay', type=float, default=5e-5)

    argumentParser.add_argument('--cae_num_epochs', type=int, default=100)
    argumentParser.add_argument('--cae_lr', type=float, default=1e-5)
    argumentParser.add_argument('--cae_weight_decay', type=float, default=5e-5)

    argumentParser.add_argument('--levels', nargs='+', type=str, default=('level_2_1', 'level_2_2', 'level_3_1', 'level_3_2', 'level_3_3', 'level_3_4', 'level_4_1', 'level_4_2', 'level_4_3', 'level_4_4'))
    argumentParser.add_argument('--pool', type=str, choices=('avgpool', 'maxpool'), default='avgpool')
    argumentParser.add_argument('--padding_mode', type=str, choices=('zeros', 'reflect', 'replicate', 'circular'), default='reflect')

    argumentParser.add_argument('--gamma', type=int, default=4)

    argumentParser.add_argument('--alpha', type=int, default=3)
    argumentParser.add_argument('--betas', nargs='+', type=int, default=(2, 2, 2))

    argumentParser.add_argument('--eta', nargs='+', type=int, default=(8, 8))
    argumentParser.add_argument('--sigma', nargs='+', type=int, default=(4, 4))

    argumentParser.add_argument('--dataset', type=str, choices=('mvtec', 'btad', 'visa'), default='mvtec')

    argumentParser.add_argument('--categories', nargs='+', type=str, default=('tile', 'wood', 'cable', 'metal_nut', 'transistor'))
    argumentParser.add_argument('--weights', nargs='+', type=list, default=((8, 4, 1), (8, 1, 1), (1, 4, 8), (8, 4, 1), (1, 4, 8)))
    # argumentParser.add_argument('--categories', nargs='+', type=str, default=('01', '02', '03'))

    argumentParser.add_argument('--data_path', type=str, default='data')
    argumentParser.add_argument('--pretrain_path', type=str, default='pretrain')
    argumentParser.add_argument('--result_path', type=str, default='result')

    argumentParser.add_argument('--evaluate_interval', type=int, default=1)

    argumentParser.add_argument('--expect_fprs', nargs='+', type=float, default=(0.0001, 0.0005, 0.001))

    args = argumentParser.parse_args()

    settings = {}
    settings['plus'] = args.plus
    settings['device'] = device('cpu') if args.device == -1 else device('cuda:{}'.format(args.device))
    settings['num_workers'] = args.num_workers
    settings['image_size'] = args.image_size
    settings['levels'] = args.levels
    settings['pool'] = args.pool
    settings['padding_mode'] = args.padding_mode
    settings['gamma'] = args.gamma
    settings['alpha'] = args.alpha
    settings['betas'] = args.betas
    settings['eta'] = args.eta
    settings['sigma'] = args.sigma
    settings['dataset'] = args.dataset
    settings['categories'] = args.categories
    settings['weights'] = args.weights
    settings['data_path'] = args.data_path
    settings['pretrain_path'] = args.pretrain_path

    if args.mode == 'train':
        settings['batch_size'] = args.batch_size
        settings['iaff'] = {}
        settings['iaff']['num_epochs'] = args.iaff_num_epochs
        settings['iaff']['lr'] = args.iaff_lr
        settings['iaff']['weight_decay'] = args.iaff_weight_decay
        settings['cae'] = {}
        settings['cae']['num_epochs'] = args.cae_num_epochs
        settings['cae']['lr'] = args.cae_lr
        settings['cae']['weight_decay'] = args.cae_weight_decay
        settings['evaluate_interval'] = args.evaluate_interval
        train(settings=settings)
    elif args.mode == 'test':
        settings['result_path'] = args.result_path
        settings['expect_fprs'] = args.expect_fprs
        test(settings=settings)
