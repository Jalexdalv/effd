from backbone.feature_extractor import Vgg19FeatureExtractor
from evaluate import compute_threshold, segment
from data import btad, mvtec, visa
from model.cae import CAE
from model.dfr import DFR, DFRPlus
from model.iaff import IAFF, IAFFPlus
from os import listdir
from os.path import join
from torch import load
from torch.nn import ModuleList


def test(settings: dict) -> None:
    dataset_path = join(settings['data_path'], settings['dataset'])

    if settings['plus']:
        for category, weight in zip(settings['categories'], settings['weights']):
            print('========================category：{}  weight：{}========================'.format(category, weight))
            category_path = join(dataset_path, category)
            pretrain_path = join(settings['pretrain_path'], category)
            result_path = join(settings['result_path'], category)

            if settings['dataset'] == 'mvtec':
                train_dataset = mvtec.TrainDataset(category_path=category_path, batch_size=1, num_workers=settings['num_workers'], image_size=settings['image_size'])
                test_dataset = mvtec.TestDataset(category_path=category_path, num_workers=settings['num_workers'], image_size=settings['image_size'])
            elif settings['dataset'] == 'btad':
                train_dataset = btad.TrainDataset(category_path=category_path, batch_size=1, num_workers=settings['num_workers'], image_size=settings['image_size'])
                test_dataset = btad.TestDataset(category_path=category_path, num_workers=settings['num_workers'], image_size=settings['image_size'])
            else:
                train_dataset = visa.TrainDataset(category_path=category_path, batch_size=1, num_workers=settings['num_workers'], image_size=settings['image_size'])
                test_dataset = visa.TestDataset(category_path=category_path, num_workers=settings['num_workers'], image_size=settings['image_size'])

            feature_extractor = Vgg19FeatureExtractor(levels=settings['levels'], pool=settings['pool'], padding_mode=settings['padding_mode']).to(device=settings['device']).eval()

            for pretrain_model in listdir(path=pretrain_path):
                iaffs = ModuleList([ModuleList([IAFFPlus(in_channels=num_channel_i_i, gamma=settings['gamma']) for num_channel_i_i in num_channels_i[0:-1]]) for num_channels_i in feature_extractor.num_channels]).to(device=settings['device']).eval()
                iaffs.load_state_dict(state_dict=load(f=join(pretrain_path, 'iaffs.pth')))
                cae = CAE(in_channels=sum([num_channels_i[0] for num_channels_i in feature_extractor.num_channels]), alpha=settings['alpha'], betas=settings['betas']).to(device=settings['device']).eval()
                cae.load_state_dict(state_dict=load(f=join(pretrain_path, 'cae.pth')))
                dfr = DFRPlus(feature_extractor=feature_extractor, iaffs=iaffs, cae=cae, weight=weight, image_size=settings['image_size'], eta=settings['eta'], sigma=settings['sigma']).to(device=settings['device']).eval()

                thresholds = compute_threshold(model=dfr, train_dataset=train_dataset, expect_fprs=settings['expect_fprs'])
                segment(model=dfr, test_dataset=test_dataset, thresholds=thresholds, result_path=join(result_path, pretrain_model))
    else:
        for category in settings['categories']:
            print('========================category：{}========================'.format(category))
            category_path = join(dataset_path, category)
            pretrain_path = join(settings['pretrain_path'], category)
            result_path = join(settings['result_path'], category)

            if settings['dataset'] == 'mvtec':
                train_dataset = mvtec.TrainDataset(category_path=category_path, batch_size=1, num_workers=settings['num_workers'], image_size=settings['image_size'])
                test_dataset = mvtec.TestDataset(category_path=category_path, num_workers=settings['num_workers'], image_size=settings['image_size'])
            elif settings['dataset'] == 'btad':
                train_dataset = btad.TrainDataset(category_path=category_path, batch_size=1, num_workers=settings['num_workers'], image_size=settings['image_size'])
                test_dataset = btad.TestDataset(category_path=category_path, num_workers=settings['num_workers'], image_size=settings['image_size'])
            else:
                train_dataset = visa.TrainDataset(category_path=category_path, batch_size=1, num_workers=settings['num_workers'], image_size=settings['image_size'])
                test_dataset = visa.TestDataset(category_path=category_path, num_workers=settings['num_workers'], image_size=settings['image_size'])

            feature_extractor = Vgg19FeatureExtractor(levels=settings['levels'], pool=settings['pool'], padding_mode=settings['padding_mode']).to(device=settings['device']).eval()

            for pretrain_model in listdir(path=pretrain_path):
                iaffs = ModuleList([ModuleList([IAFF(in_channels=num_channel_i_i, gamma=settings['gamma']) for num_channel_i_i in num_channels_i[0:-1]]) for num_channels_i in feature_extractor.num_channels]).to(device=settings['device']).eval()
                iaffs.load_state_dict(state_dict=load(f=join(pretrain_path, 'iaffs.pth')))
                cae = CAE(in_channels=sum([num_channels_i[0] for num_channels_i in feature_extractor.num_channels]), alpha=settings['alpha'], betas=settings['betas']).to(device=settings['device']).eval()
                cae.load_state_dict(state_dict=load(f=join(pretrain_path, 'cae.pth')))
                dfr = DFR(feature_extractor=feature_extractor, iaffs=iaffs, cae=cae, image_size=settings['image_size'], eta=settings['eta'], sigma=settings['sigma']).to(device=settings['device']).eval()

                thresholds = compute_threshold(model=dfr, train_dataset=train_dataset, expect_fprs=settings['expect_fprs'])
                segment(model=dfr, test_dataset=test_dataset, thresholds=thresholds, result_path=join(result_path, pretrain_model))
