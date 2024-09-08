from backbone.feature_extractor import Vgg19FeatureExtractor
from evaluate import compute_auc_roc
from data import btad, mvtec, visa
from model.cae import CAE
from model.dfr import DFR, DFRPlus
from model.effd import EFFD
from model.iaff import IAFF, IAFFPlus
from os.path import exists, join
from torch import load, save
from torch.cuda import empty_cache
from torch.nn import ModuleList
from torch.optim import AdamW
from tqdm import tqdm
from utils import create_dir, load_from_pickle, save_to_pickle


def train(settings: dict) -> None:
    dataset_path = join(settings['data_path'], settings['dataset'])

    if settings['plus']:
        for category, weight in zip(settings['categories'], settings['weights']):
            print('========================category：{}  weight：{}========================'.format(category, weight))
            category_path = join(dataset_path, category)
            pretrain_path = join(settings['pretrain_path'], category)
            create_dir(path=pretrain_path)

            if settings['dataset'] == 'mvtec':
                train_dataset = mvtec.TrainDataset(category_path=category_path, batch_size=settings['batch_size'], num_workers=settings['num_workers'], image_size=settings['image_size'])
                test_dataset = mvtec.TestDataset(category_path=category_path, num_workers=settings['num_workers'], image_size=settings['image_size'])
            elif settings['dataset'] == 'btad':
                train_dataset = btad.TrainDataset(category_path=category_path, batch_size=settings['batch_size'], num_workers=settings['num_workers'], image_size=settings['image_size'])
                test_dataset = btad.TestDataset(category_path=category_path, num_workers=settings['num_workers'], image_size=settings['image_size'])
            else:
                train_dataset = visa.TrainDataset(category_path=category_path, batch_size=settings['batch_size'], num_workers=settings['num_workers'], image_size=settings['image_size'])
                test_dataset = visa.TestDataset(category_path=category_path, num_workers=settings['num_workers'], image_size=settings['image_size'])

            feature_extractor = Vgg19FeatureExtractor(levels=settings['levels'], pool=settings['pool'], padding_mode=settings['padding_mode']).to(device=settings['device']).eval()

            effd = EFFD(feature_extractor=feature_extractor, image_size=settings['image_size'], eta=settings['eta']).to(device=settings['device'])

            distribution_path = join(pretrain_path, 'distributions.pkl')
            if not exists(path=distribution_path):
                print('============estimating and fusing feature distribution============')
                sample_features = effd.get_sample_features(train_dataset=train_dataset)
                print('extracting completed')
                distributions = effd(sample_features=sample_features)
                save_to_pickle(object=distributions, path=distribution_path)
                print('saving completed')
            else:
                distributions = load_from_pickle(path=distribution_path)
                print('distributions loaded')

            iaffs = ModuleList([ModuleList([IAFFPlus(in_channels=num_channel_i_i, gamma=settings['gamma']) for num_channel_i_i in num_channels_i[0:-1]]) for num_channels_i in feature_extractor.num_channels]).to(device=settings['device'])
            cae = CAE(in_channels=sum([num_channels_i[0] for num_channels_i in feature_extractor.num_channels]), alpha=settings['alpha'], betas=settings['betas']).to(device=settings['device'])
            dfr = DFRPlus(feature_extractor=feature_extractor, iaffs=iaffs, cae=cae, weight=weight, image_size=settings['image_size'], eta=settings['eta'], sigma=settings['sigma']).to(device=settings['device'])

            # training iaff
            iaffs_path = join(pretrain_path, 'iaffs.pth')
            if not exists(path=iaffs_path):
                optimizer = AdamW(params=iaffs.parameters(), lr=settings['iaff']['lr'], weight_decay=settings['iaff']['weight_decay'])
                print('============training iaff============')
                for index in range(1, settings['iaff']['num_epochs'] + 1):
                    print('epoch：{}'.format(index))
                    loss_sum, num_samples, auc_roc = 0, 0, 0
                    with tqdm(iterable=train_dataset.dataloader, unit='batch') as batches:
                        for batch in batches:
                            loss = dfr.compute_distribution_loss(input=batch.to(device=settings['device']), distributions=distributions)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            loss_sum += loss.item()
                            num_samples += batch.shape[0]
                    print('loss：{}'.format(loss_sum / num_samples))
                save(obj=iaffs.state_dict(), f=iaffs_path)
                print('saving completed')
            else:
                iaffs.load_state_dict(state_dict=load(f=iaffs_path))
                print('iaffs loaded')

            # training cae
            print('============training cae============')
            best_auc_roc = 0
            optimizer = AdamW(params=cae.parameters(), lr=settings['cae']['lr'], weight_decay=settings['cae']['weight_decay'])
            for index in range(1, settings['cae']['num_epochs'] + 1):
                print('epoch：{}'.format(index))
                loss_sum, num_samples, auc_roc = 0, 0, 0
                with tqdm(iterable=train_dataset.dataloader, unit='batch') as batches:
                    for batch in batches:
                        loss = dfr.compute_reconstruction_loss(input=batch.to(device=settings['device']))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        loss_sum += loss.item()
                        num_samples += batch.shape[0]
                print('loss：{}'.format(loss_sum / num_samples))

                if index % settings['evaluate_interval'] == 0:
                    dfr.eval()
                    auc_roc = compute_auc_roc(model=dfr, test_dataset=test_dataset)
                    dfr.train()
                    if auc_roc > best_auc_roc:
                        best_auc_roc = auc_roc
                        save(obj=cae.state_dict(), f=join(pretrain_path, 'cae-{}.pth'.format(auc_roc)))
                        print('saving completed')
            empty_cache()

    else:
        for category in settings['categories']:
            print('========================category：{}========================'.format(category))
            category_path = join(dataset_path, category)
            pretrain_path = join(settings['pretrain_path'], category)
            create_dir(path=pretrain_path)

            if settings['dataset'] == 'mvtec':
                train_dataset = mvtec.TrainDataset(category_path=category_path, batch_size=settings['batch_size'], num_workers=settings['num_workers'], image_size=settings['image_size'])
                test_dataset = mvtec.TestDataset(category_path=category_path, num_workers=settings['num_workers'], image_size=settings['image_size'])
            elif settings['dataset'] == 'btad':
                train_dataset = btad.TrainDataset(category_path=category_path, batch_size=settings['batch_size'], num_workers=settings['num_workers'], image_size=settings['image_size'])
                test_dataset = btad.TestDataset(category_path=category_path, num_workers=settings['num_workers'], image_size=settings['image_size'])
            else:
                train_dataset = visa.TrainDataset(category_path=category_path, batch_size=settings['batch_size'], num_workers=settings['num_workers'], image_size=settings['image_size'])
                test_dataset = visa.TestDataset(category_path=category_path, num_workers=settings['num_workers'], image_size=settings['image_size'])

            feature_extractor = Vgg19FeatureExtractor(levels=settings['levels'], pool=settings['pool'], padding_mode=settings['padding_mode']).to(device=settings['device']).eval()

            effd = EFFD(feature_extractor=feature_extractor, image_size=settings['image_size'], eta=settings['eta']).to(device=settings['device'])

            distribution_path = join(pretrain_path, 'distributions.pkl')
            if not exists(path=distribution_path):
                print('============estimating and fusing feature distribution============')
                sample_features = effd.get_sample_features(train_dataset=train_dataset)
                print('extracting completed')
                distributions = effd(sample_features=sample_features)
                save_to_pickle(object=distributions, path=distribution_path)
                print('saving completed')
            else:
                distributions = load_from_pickle(path=distribution_path)
                print('distributions loaded')

            iaffs = ModuleList([ModuleList([IAFF(in_channels=num_channel_i_i, gamma=settings['gamma']) for num_channel_i_i in num_channels_i[0:-1]]) for num_channels_i in feature_extractor.num_channels]).to(device=settings['device'])
            cae = CAE(in_channels=sum([num_channels_i[0] for num_channels_i in feature_extractor.num_channels]), alpha=settings['alpha'], betas=settings['betas']).to(device=settings['device'])
            dfr = DFR(feature_extractor=feature_extractor, iaffs=iaffs, cae=cae, image_size=settings['image_size'], eta=settings['eta'], sigma=settings['sigma']).to(device=settings['device'])

            # training iaff
            iaffs_path = join(pretrain_path, 'iaffs.pth')
            if not exists(path=iaffs_path):
                optimizer = AdamW(params=iaffs.parameters(), lr=settings['iaff']['lr'], weight_decay=settings['iaff']['weight_decay'])
                print('============training iaff============')
                for index in range(1, settings['iaff']['num_epochs'] + 1):
                    print('epoch：{}'.format(index))
                    loss_sum, num_samples, auc_roc = 0, 0, 0
                    with tqdm(iterable=train_dataset.dataloader, unit='batch') as batches:
                        for batch in batches:
                            loss = dfr.compute_distribution_loss(input=batch.to(device=settings['device']), distributions=distributions)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            loss_sum += loss.item()
                            num_samples += batch.shape[0]
                    print('loss：{}'.format(loss_sum / num_samples))
                save(obj=iaffs.state_dict(), f=iaffs_path)
                print('saving completed')
            else:
                iaffs.load_state_dict(state_dict=load(f=iaffs_path))
                print('iaffs loaded')

            # training cae
            print('============training cae============')
            best_auc_roc = 0
            optimizer = AdamW(params=cae.parameters(), lr=settings['cae']['lr'], weight_decay=settings['cae']['weight_decay'])
            for index in range(1, settings['cae']['num_epochs'] + 1):
                print('epoch：{}'.format(index))
                loss_sum, num_samples, auc_roc = 0, 0, 0
                with tqdm(iterable=train_dataset.dataloader, unit='batch') as batches:
                    for batch in batches:
                        loss = dfr.compute_reconstruction_loss(input=batch.to(device=settings['device']))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        loss_sum += loss.item()
                        num_samples += batch.shape[0]
                print('loss：{}'.format(loss_sum / num_samples))

                if index % settings['evaluate_interval'] == 0:
                    dfr.eval()
                    auc_roc = compute_auc_roc(model=dfr, test_dataset=test_dataset)
                    dfr.train()
                    if auc_roc > best_auc_roc:
                        best_auc_roc = auc_roc
                        save(obj=cae.state_dict(), f=join(pretrain_path, 'cae-{}.pth'.format(auc_roc)))
                        print('saving completed')
            empty_cache()
