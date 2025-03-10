import torch
import numpy as np
from model import Transformer_ST, Model_all, ScoreMatch_module, SMASH
from torch.optim import AdamW
import argparse
from model.Dataset import get_dataloader
from model.Metric import get_calibration_score
import time
import datetime
import pickle
import os
from tqdm import tqdm
import random
import sys
import math
import pandas as pd
import json

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Your code here

def setup_init(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def model_name():
    TIME = int(time.time())
    TIME = time.localtime(TIME)
    return time.strftime("%Y-%m-%d %H:%M:%S",TIME)

def normalization(x,MAX,MIN):
    return (x-MIN)/(MAX-MIN)

def denormalization(x,MAX,MIN,log_normalization=False):
    if log_normalization:
        return torch.exp(x.detach().cpu()*(MAX-MIN)+MIN)
    else:
        return x.detach().cpu()*(MAX-MIN)+MIN

def get_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--seed', type=int, default=1234, help='')
    parser.add_argument('--model', type=str, default='SMASH', help='')
    parser.add_argument('--seq_len', type=int, default = 100, help='')
    parser.add_argument('--mode', type=str, default='train', help='')
    parser.add_argument('--total_epochs', type=int, default=1000, help='')
    parser.add_argument('--machine', type=str, default='none', help='')
    parser.add_argument('--dim', type=int, default=3, help='', choices = [1,2,3])
    parser.add_argument('--dataset', type=str, default='Earthquake',choices=['Earthquake','crime','football','ComCat','WHITE','SCEDC','SaltonSea','SanJac'], help='')
    parser.add_argument('--batch_size', type=int, default=8,help='')
    parser.add_argument('--samplingsteps', type=int, default=500, help='')
    parser.add_argument('--per_step', type=int, default=250, help='')
    parser.add_argument('--cuda_id', type=str, default='0', help='')
    parser.add_argument('--n_samples', type=int, default=100, help='')
    parser.add_argument('--log_normalization', type=int, default=1, help='')
    parser.add_argument('--weight_path', type=str, default='./ModelSave/dataset_Earthquake_model_SMSTPP/model_300.pkl', help='')
    parser.add_argument('--save_path', type=str, help='')
    parser.add_argument('--cond_dim', type=int, default=64, help='')
    parser.add_argument('--sigma_time', type=float, default=0.05, help='')
    parser.add_argument('--sigma_loc', type=float, default=0.05, help='')
    parser.add_argument('--langevin_step', type=float, default=0.005, help='')
    parser.add_argument('--loss_lambda', type=float, default=0.5, help='')
    parser.add_argument('--loss_lambda2', type=float, default=1, help='')
    parser.add_argument('--smooth', type=float, default=0.0, help='')
    parser.add_argument('--Mcut', type=float, help='')
    parser.add_argument('--catalog_path', type=str, help='')
    parser.add_argument('--auxiliary_start', type=lambda s : pd.to_datetime(s,format='%Y-%m-%d:%H:%M:%S'), help='')
    parser.add_argument('--train_nll_start', type=lambda s : pd.to_datetime(s,format='%Y-%m-%d:%H:%M:%S'), help='')
    parser.add_argument('--val_nll_start', type=lambda s : pd.to_datetime(s,format='%Y-%m-%d:%H:%M:%S'), help='')
    parser.add_argument('--test_nll_start', type=lambda s : pd.to_datetime(s,format='%Y-%m-%d:%H:%M:%S'), help='')
    parser.add_argument('--test_nll_end', type=lambda s : pd.to_datetime(s,format='%Y-%m-%d:%H:%M:%S'), help='')
    parser.add_argument('--marked_output', type=int, default=1, help='')
    parser.add_argument('--num_catalogs', type=int, default=5000, help='')
    parser.add_argument('--day_number', type=int, default=0, help='')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print(args)  # Add this line to print the arguments
    return args

opt = get_args()
device = torch.device("cuda:{}".format(opt.cuda_id) if opt.cuda else "cpu")
# device = torch.device("cpu")

if opt.dataset == 'HawkesGMM':
    opt.dim=1

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda_id)

def process_split(opt,df,pkl_filename):

    data_array = df.to_numpy()
    num_batches = len(data_array) // opt.seq_len
    batches = np.array_split(data_array[:num_batches * opt.seq_len], num_batches)

    batches_list = []
    for batch in batches:
        # Subtract the start time from all times in the batch
        start_time = batch[0, 0]-1  # The first time in the batch
        batch[:, 0] -= start_time  # Subtract start_time from all time values in the batch
        
        # Convert the batch to a list of lists
        batches_list.append(batch.tolist())

    leftover_rows = data_array[num_batches * opt.seq_len:]
    if len(leftover_rows) > 0:
        start_time = leftover_rows[0, 0]-1
        leftover_rows[:, 0] -= start_time
        batches_list.append(leftover_rows.tolist())


    with open(pkl_filename, 'wb') as file:
        pickle.dump(batches_list, file)

def preprocess_catalog(opt):

    df = pd.read_csv(
                    opt.catalog_path,
                    parse_dates=["time"],
                    dtype={"url": str, "alert": str},
                )
    df = df.sort_values(by='time')

    df = df[['time','magnitude','x','y']]


    ### filter events by magnitude threshold

    df = df[df['magnitude']>=opt.Mcut]

    ### create train/val/test dfs
    aux_df = df[df['time']>=opt.auxiliary_start]
    aux_df = df[df['time']<opt.train_nll_start]

    # train_df = df[df['time']>=opt.train_nll_start]
    train_df = df[df['time']>=opt.auxiliary_start]
    train_df = train_df[train_df['time']< opt.val_nll_start]

    val_df = df[df['time']>=opt.val_nll_start]
    val_df = val_df[val_df['time']< opt.test_nll_start]

    test_df = df[df['time']>=opt.test_nll_start]
    test_df = test_df[test_df['time']< opt.test_nll_end]


    ## convert datetime to days

    train_df['time'] = (train_df['time']-train_df['time'].min()).dt.total_seconds() / (60*60*24)
    val_df['time'] = (val_df['time']-val_df['time'].min()).dt.total_seconds() / (60*60*24)
    test_df['time'] = (test_df['time']-test_df['time'].min()).dt.total_seconds() / (60*60*24)

    # List of DataFrames
    dfs = [train_df, val_df, test_df]

    # Process each DataFrame
    for i, df in enumerate(dfs):
        time_diffs = np.ediff1d(df['time'])

        # Identify the indices where the differences are less than or equal to 0
        indices_to_drop = np.where(time_diffs <= 0)[0] + 1

        indices_to_drop = df.index[indices_to_drop]

        # Drop the rows with the identified indices
        dfs[i] = df.drop(index=indices_to_drop)

    # Assign the processed DataFrames back
    train_df, val_df, test_df = dfs

    assert (np.ediff1d(train_df['time']) > 0).all()
    assert (np.ediff1d(val_df['time']) > 0).all()
    assert (np.ediff1d(test_df['time']) > 0).all()

    process_split(opt,train_df,'dataset/{}/seq_len_{}_mag_data_train.pkl'.format(opt.dataset,opt.seq_len))
    process_split(opt,val_df,'dataset/{}/seq_len_{}_mag_data_val.pkl'.format(opt.dataset,opt.seq_len))
    process_split(opt,test_df,'dataset/{}/seq_len_{}_mag_data_test.pkl'.format(opt.dataset,opt.seq_len))

def deg2rad(deg):
    return deg * np.pi / 180

def rad2deg(rad):
    return rad * 180 / np.pi

def azimuthal_equidistant_inverse(x, y, lat0, lon0, R=6371):
    """
    Inverse azimuthal equidistant projection.
    Converts (x, y) back to (lat, lon) from a center (lat0, lon0).
    
    Parameters:
    - x, y: Projected coordinates (km)
    - lat0, lon0: Center of the projection (degrees)
    - R: Radius of the sphere (default: Earth radius in km)
    
    Returns:
    - lat, lon: Geographic coordinates (degrees)
    """
    lat0, lon0 = deg2rad(lat0), deg2rad(lon0)
    
    r = np.sqrt(x**2 + y**2)
    c = r / R
    
    lat = np.where(r == 0, lat0, np.arcsin(np.cos(c) * np.sin(lat0) + (x * np.sin(c) * np.cos(lat0) / r)))
    lon = lon0 + np.arctan2(y * np.sin(c), r * np.cos(lat0) * np.cos(c) - x * np.sin(lat0) * np.sin(c))
    
    return rad2deg(lat), rad2deg(lon)


def create_test_day_dataloader(opt, day_number=opt.day_number, Max=None, Min=None, batch_size=32):
    df = pd.read_csv(
                    opt.catalog_path,
                    parse_dates=["time"],
                    dtype={"url": str, "alert": str},
                )
    df = df.sort_values(by='time')
    df = df[df['magnitude'] >= opt.Mcut]
    center_latitude = df['latitude'].mean()
    center_longitude = df['longitude'].mean()
    df = df[['time','magnitude','x','y']]


    test_day_begin = opt.test_nll_start + pd.Timedelta(days=day_number)
    print('test_day_begin', test_day_begin)
    test_day_end = opt.test_nll_start + pd.Timedelta(days=day_number+1)
    print('test_day_end', test_day_end)
    test_day_df = df[df['time'] < test_day_begin]

    # Convert Timestamps to numeric days
    test_day_df['time'] = (test_day_df['time'] - test_day_df['time'].min()).dt.total_seconds() / (60*60*24)
    test_day_begin = (test_day_begin - df['time'].min()).total_seconds() / (60*60*24)
    test_day_end = (test_day_end - df['time'].min()).total_seconds() / (60*60*24)

    test_day_array = test_day_df.to_numpy()


    # Keep only the last seq_len rows if needed
    if len(test_day_array) > opt.seq_len:
        test_day_array = test_day_array[-opt.seq_len:]

    # Convert Timestamps to numeric days
    start_time = test_day_array[0, 0]  

    start_time_datetime = df['time'].min() + pd.Timedelta(days=start_time-1)

    print('start_time_datetime', start_time_datetime)

    test_day_array[:, 0] = ((test_day_array[:, 0] - start_time) + 1.0)
    start_time_float = (test_day_begin - start_time) + 1.0
    end_time_float = (test_day_end - start_time) + 1.0
    print('end_time_float', end_time_float)

    print('end_datetime', start_time_datetime + pd.Timedelta(days=end_time_float))

    # convert to list of lists
    test_day_array = [test_day_array.tolist()]*opt.num_catalogs

    if not opt.log_normalization:
        test_day_data = [[[i[0], i[0]-u[index-1][0] if index>0 else i[0], i[1]+1]+ i[2:] for index, i in enumerate(u)] for u in test_day_array]
    else:
        test_day_data = [[[i[0], math.log(max(i[0]-u[index-1][0],1e-4)) if index>0 else math.log(max(i[0],1e-4)), i[1]+1]+ i[2:] for index, i in enumerate(u)] for u in test_day_array]

    test_day_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in test_day_data]

    test_dayloader = get_dataloader(test_day_data, batch_size=batch_size, D=opt.dim, shuffle=False)
    print('Min & Max', (Max, Min), opt.num_types)
    return test_dayloader, start_time_datetime, start_time_float, end_time_float, center_latitude, center_longitude


def data_loader(opt):

    f = open('dataset/{}/seq_len_{}_mag_data_train.pkl'.format(opt.dataset,opt.seq_len),'rb')
    train_data = pickle.load(f)
    train_data = [[list(i) for i in u] for u in train_data]
    f = open('dataset/{}/seq_len_{}_mag_data_val.pkl'.format(opt.dataset,opt.seq_len),'rb')
    val_data = pickle.load(f)
    val_data = [[list(i) for i in u] for u in val_data]
    f = open('dataset/{}/seq_len_{}_mag_data_test.pkl'.format(opt.dataset,opt.seq_len),'rb')
    test_data = pickle.load(f)
    test_data = [[list(i) for i in u] for u in test_data]

    if not opt.log_normalization:
        train_data = [[[i[0], i[0]-u[index-1][0] if index>0 else i[0], i[1]+1]+ i[2:] for index, i in enumerate(u)] for u in train_data]
        val_data = [[[i[0], i[0]-u[index-1][0] if index>0 else i[0], i[1]+1]+ i[2:] for index, i in enumerate(u)] for u in val_data]
        test_data = [[[i[0], i[0]-u[index-1][0] if index>0 else i[0], i[1]+1]+ i[2:] for index, i in enumerate(u)] for u in test_data]
    else:
        train_data = [[[i[0], math.log(max(i[0]-u[index-1][0],1e-4)) if index>0 else math.log(max(i[0],1e-4)), i[1]+1]+ i[2:] for index, i in enumerate(u)] for u in train_data]
        val_data = [[[i[0], math.log(max(i[0]-u[index-1][0],1e-4)) if index>0 else math.log(max(i[0],1e-4)), i[1]+1]+ i[2:] for index, i in enumerate(u)] for u in val_data]
        test_data = [[[i[0], math.log(max(i[0]-u[index-1][0],1e-4)) if index>0 else math.log(max(i[0],1e-4)), i[1]+1]+ i[2:] for index, i in enumerate(u)] for u in test_data]

    data_all = train_data+test_data+val_data

    Max, Min = [], []
    for m in range(opt.dim+2):
        if m > 0:
            Max.append(max([i[m] for u in data_all for i in u]))
            Min.append(min([i[m] for u in data_all for i in u]))
        else:
            Max.append(1)
            Min.append(0)
        
    if opt.dim==3:
        Max[2] = 1
        Min[2] = 0
        opt.num_types=int(max([i[2] for u in data_all for i in u])) 
    else:
        opt.num_types = 1

    print('num_types:', opt.num_types)
    
    train_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in train_data]
    test_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in test_data]
    val_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in val_data]
    trainloader = get_dataloader(train_data, opt.batch_size, D = opt.dim, shuffle=True)
    testloader = get_dataloader(test_data, opt.batch_size, D = opt.dim, shuffle=False)
    valloader = get_dataloader(test_data, opt.batch_size, D = opt.dim, shuffle=False)
    print('Min & Max', (Max, Min), opt.num_types)
    return trainloader, testloader, valloader, (Max,Min)


def Batch2toModel(batch, transformer):

    if opt.dim==2:
        event_time_origin, event_time, lng, lat = map(lambda x: x.to(device), batch)
        event_loc = torch.cat((lng.unsqueeze(dim=2),lat.unsqueeze(dim=2)),dim=-1)

    if opt.dim==3:
        event_time_origin, event_time, mark, lng, lat = map(lambda x: x.to(device), batch)

        event_loc = torch.cat((mark.unsqueeze(dim=2), lng.unsqueeze(dim=2),lat.unsqueeze(dim=2)),dim=-1)

    event_time = event_time.to(device)
    event_time_origin = event_time_origin.to(device)
    event_loc = event_loc.to(device)
    
    enc_out, mask = transformer(event_loc, event_time_origin)

    # print(event_time.size(),event_loc.size(), enc_out.size(),mask.size())
    enc_out_non_mask  = []
    event_time_non_mask = []
    event_loc_non_mask = []
    for index in range(mask.shape[0]):
        length = int(sum(mask[index]).item())

        if length>1:
            enc_out_non_mask += [i.unsqueeze(dim=0) for i in enc_out[index][:length-1]]
            event_time_non_mask += [i.unsqueeze(dim=0) for i in event_time[index][1:length]]
            event_loc_non_mask += [i.unsqueeze(dim=0) for i in event_loc[index][1:length]]


    enc_out_non_mask = torch.cat(enc_out_non_mask,dim=0)
    event_time_non_mask = torch.cat(event_time_non_mask,dim=0)
    event_loc_non_mask = torch.cat(event_loc_non_mask,dim=0)

    event_time_non_mask = event_time_non_mask.reshape(-1,1,1)
    event_loc_non_mask = event_loc_non_mask.reshape(-1,1,opt.dim)
    
    enc_out_non_mask = enc_out_non_mask.reshape(event_time_non_mask.shape[0],1,-1)
    return event_time_non_mask, event_loc_non_mask, enc_out_non_mask


def LR_warmup(lr, epoch_num, epoch_current):
    return lr * (epoch_current+1) / epoch_num


if __name__ == "__main__":
    
    setup_init(opt)

    print('dataset:{}'.format(opt.dataset))
    from datetime import datetime
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

    model_path = opt.save_path


    if not os.path.exists('./ModelSave'):
        os.mkdir('./ModelSave')

    if 'train' in opt.mode and not os.path.exists(model_path):
        os.mkdir(model_path)

    preprocess_catalog(opt)

    trainloader, testloader, valloader, (MAX,MIN) = data_loader(opt)

    if not opt.marked_output:
        opt.loss_lambda = 0
        print('-----------Unmarked Output-----------------')

    model= ScoreMatch_module(
        dim=1+opt.dim,
        condition = True,
        cond_dim=opt.cond_dim,
        num_types=opt.num_types
    ).to(device)

    decoder = SMASH(
        model,
        sigma = (opt.sigma_time,opt.sigma_loc),
        seq_length = 1+opt.dim,
        sampling_timesteps = opt.samplingsteps,
        n_samples=opt.n_samples,
        langevin_step=opt.langevin_step,
        num_types=opt.num_types,
        loss_lambda = opt.loss_lambda,
        loss_lambda2 = opt.loss_lambda2,
        smooth=opt.smooth,
        device=device
    ).to(device)

    transformer = Transformer_ST(
        d_model=opt.cond_dim,
        d_rnn=opt.cond_dim*4,
        d_inner=opt.cond_dim*2,
        n_layers=4,
        n_head=4,
        d_k=16,
        d_v=16,
        dropout=0.1,
        device=device,
        loc_dim = opt.dim,
        CosSin = True,
        num_types=opt.num_types
    ).to(device)

    Model = Model_all(transformer,decoder)
    if opt.mode == 'test' or opt.mode == 'sample':
        Model.load_state_dict(torch.load(opt.weight_path))
        print('Weight loaded!!')
    total_params = sum(p.numel() for p in Model.parameters())
    print(f"Number of parameters: {total_params}")

    warmup_steps = 5
    # training
    optimizer = AdamW(Model.parameters(), lr = 1e-3, betas = (0.9, 0.99))
    step, early_stop = 0, 0
    min_loss_test = 1e20
    for itr in tqdm(range(opt.total_epochs)):

        print('epoch:{}'.format(itr))

        if (itr % 10==0 ) or (opt.mode == 'test') or (opt.mode == 'sample'):
            
            Model.eval()

            if opt.mode == 'test':
            
                print('Evaluate!')

                # testing set
                loss_test_all = 0.0         
                for batch in testloader:
                    event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)
                    loss = Model.decoder(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1), enc_out_non_mask)
                    loss_test_all += loss.item() * event_time_non_mask.shape[0]

                current_step = 0
                is_last = False
                last_sample = None
                while current_step < opt.samplingsteps:
                    if (current_step + opt.per_step) >= opt.samplingsteps:
                        is_last = True
                    cs_time_all = torch.zeros(5)
                    cs_loc_all = torch.zeros(5)
                    cs2_time_all = torch.zeros(5)
                    cs2_loc_all = torch.zeros(5)
                    acc_all = 0
                    ece_all = 0
                    correct_list_all = torch.zeros(10)
                    num_list_all = torch.zeros(10)
                    mae_temporal, mae_spatial, total_num = 0.0, 0.0, 0.0
                    sampled_record_all = []
                    gt_record_all = []

                    for idx, batch in enumerate(testloader):

                        event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)


                        real_time = denormalization(event_time_non_mask[:,0,:], MAX[1], MIN[1], opt.log_normalization)
                        real_loc = event_loc_non_mask[:,0,:]
                        real_loc = denormalization(real_loc, torch.tensor([MAX[2:]]), torch.tensor([MIN[2:]]))

                        total_num += real_loc.shape[0]
                        gt_record_all.append(torch.cat((real_time,real_loc),-1))
                        
                        
                        sampled_seq_all, sampled_seq_temporal_all, sampled_seq_spatial_all, sampled_seq_mark_all = [], [], [], []
                        for i in range(int(300/opt.n_samples)):
                            sampled_seq, score_mark = Model.decoder.sample_from_last(batch_size = event_time_non_mask.shape[0],step=opt.per_step, is_last = is_last, cond=enc_out_non_mask, last_sample = last_sample[idx][i] if last_sample is not None else None)

                            print("sampled_seq.shape",sampled_seq.shape)

                            sampled_seq_all.append((sampled_seq.detach(),score_mark.detach() if score_mark is not None else None))
                            sampled_seq_temporal_all.append(denormalization(sampled_seq[:,:,0], MAX[1], MIN[1], opt.log_normalization))
                            sampled_seq_spatial_all.append(denormalization(sampled_seq[:,:,-2:], torch.tensor([MAX[-2:]]), torch.tensor([MIN[-2:]])))
                            sampled_seq_mark_all.append(score_mark.detach().cpu())
                        
                        sampled_record_all.append(sampled_seq_all)
                        gen_time = torch.cat(sampled_seq_temporal_all,1).mean(1,keepdim=True)
                        assert real_time.shape==gen_time.shape
                        mae_temporal += torch.abs(real_time-gen_time).sum().item()
                    
                        gen_loc = torch.cat(sampled_seq_spatial_all,1).mean(1)
                        assert real_loc[:,-2:].shape==gen_loc.shape
                        mae_spatial += torch.sqrt(torch.sum((real_loc[:,-2:]-gen_loc)**2,dim=-1)).sum().item()
                        
                        if score_mark is not None:
                            gen_mark = torch.mode(torch.max(torch.cat(sampled_seq_mark_all,1), dim=-1)[1],1)[0]
                            acc_all += torch.sum(gen_mark == (real_loc[:,0]-1))


                        if opt.mode=='test':
                            calibration_score = get_calibration_score(sampled_seq_temporal_all, sampled_seq_spatial_all, sampled_seq_mark_all, real_time, real_loc)
                            cs_time_all += calibration_score[0]
                            cs_loc_all += calibration_score[1]
                            cs2_time_all += calibration_score[2]
                            cs2_loc_all += calibration_score[3]
                            if score_mark is not None:
                                ece_all += calibration_score[4]
                                correct_list_all += calibration_score[5]
                                num_list_all += calibration_score[6]

                    last_sample = sampled_record_all
                    current_step += opt.per_step

                    if opt.mode=='test':
                        cs_time_all /= total_num
                        cs_loc_all /= total_num
                        cs2_time_all /= total_num
                        cs2_loc_all /= total_num
                        ece_all /= total_num
                        correct_list_all /= num_list_all
                        print('Step: ',current_step)
                        print('Calibration Score Quantile: ',cs2_time_all, cs2_loc_all)
                        print('Calibration Score: ',cs_time_all.mean().item(), cs_loc_all.mean().item())
                        print('MAE: ',mae_temporal/total_num, mae_spatial/total_num)
                        if score_mark is not None:
                            print('Mark: ',acc_all/total_num, ece_all, correct_list_all)
                    
                    global_step = itr if opt.mode=='train' else current_step

                if opt.mode=='test':
                    torch.save([sampled_record_all, gt_record_all], './samples/test_{}_{}_sigma_{}_{}_steps_{}_log_{}seq_len_{}_marked_output_{}.pkl'.format(opt.dataset, opt.model,opt.sigma_time,opt.sigma_loc ,opt.samplingsteps,opt.log_normalization,opt.seq_len,opt.marked_output))
                    with open(model_path + "results.json", "w") as f:
                        json.dump({"cs2_time_all": cs_time_all, 
                            "cs2_loc_all": cs2_loc_all, 
                            "cs2_time_mean":cs_time_all.mean().item(), 
                            "cs2_loc_mean":cs_loc_all.mean().item(),
                            "MAE_time": mae_temporal/total_num,
                            "MAE_loc": mae_spatial/total_num}, f, indent=4)
                    break


            if opt.mode == 'sample':

                test_day_loader, start_time_datetime, start_time_float, end_time_float, center_lat, center_lon = create_test_day_dataloader(opt, day_number=opt.day_number, Max=MAX, Min=MIN,batch_size=opt.batch_size)

                print('Sampling!')
                for idx, batch in enumerate(test_day_loader):
                    print('Batch {} of {}'.format(idx, len(test_day_loader)))
                    which_under_end_time = torch.ones(1, batch[0].shape[0], dtype=torch.bool)
                    # create a list of indexes that are alive based on the index within the whole test data
                    indexes_alive = list(range(idx*opt.batch_size, idx*opt.batch_size+ batch[0].shape[0]))

                    # create an empty df to store the generated events
                    gen_df = pd.DataFrame(columns=['mag','time_string','x','y','depth','catalog_id'])
                    
                    round_number = 0
                    # while at least one random sequence is whithin the forecast horizon
                    while which_under_end_time.sum() > 0:

                        round_number += 1
                        print('Generation:', round_number)
                        sampled_record_all = []
                        current_step = 0
                        is_last = False
                        last_sample = None
                    
                        while current_step < opt.samplingsteps:
                            print('Step:', current_step)
                            if (current_step + opt.per_step) >= opt.samplingsteps:
                                is_last = True
                            
                            event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)

                            # isolate last event in the batch
                            enc_out_non_mask = enc_out_non_mask[opt.seq_len-2::opt.seq_len-1,:,:]
                                                    
                            sampled_seq, score_mark = Model.decoder.sample_from_last(batch_size = enc_out_non_mask.shape[0],
                                                                                        step=opt.per_step, 
                                                                                        is_last = is_last, 
                                                                                        cond=enc_out_non_mask, 
                                                                                        last_sample = last_sample if last_sample is not None else None)

                            sampled_seq_all = (sampled_seq.detach(),score_mark.detach() if score_mark is not None else None)
                            sampled_seq_temporal_all = denormalization(sampled_seq[:,:,0], MAX[1], MIN[1], opt.log_normalization).detach()
                            sampled_seq_spatial_all = denormalization(sampled_seq[:,:,-2:], torch.tensor([MAX[-2:]]), torch.tensor([MIN[-2:]])).detach()
                            sampled_seq_mark_all = score_mark.detach()

                            last_sample = sampled_seq_all
                            current_step += opt.per_step
                            
                            global_step = itr if opt.mode=='train' else current_step


                        # convert the generated events to datetime
                        last_event_times = batch[0][:,-1]
                        gen_event_times = last_event_times + sampled_seq_temporal_all.cpu().t()
                        gen_event_datetimes = start_time_datetime + pd.to_timedelta(gen_event_times.numpy().flatten(),unit='D')
                        gen_mark = torch.mode(torch.max(sampled_seq_mark_all,dim=-1)[1],1)[0].cpu()


                        gen_events = torch.cat((gen_event_times.t().unsqueeze(dim=2), sampled_seq.cpu(), gen_mark.unsqueeze(dim=1).unsqueeze(dim=2)), dim=-1)

                        which_over_start_time = (gen_event_times > start_time_float)[0]
                        which_under_end_time = (gen_event_times < end_time_float)[0]
                        # only keep indexes which unders the end time return empty list if none
                        indexes_alive = [i for idx, i in enumerate(indexes_alive) if which_under_end_time[idx].item()]

                        # Append the generated events to the DataFrame using pd.concat
                        for idx, i in enumerate(indexes_alive):
                            if (which_under_end_time[idx].item()) and (which_over_start_time[idx].item()):
                                new_row = pd.DataFrame([{
                                    'mag': gen_mark[idx].item()+1,
                                    'time_string': gen_event_datetimes[idx].strftime('%Y-%m-%dT%H:%M:%S'),
                                    'x': sampled_seq_spatial_all[idx, 0, 0].item(),
                                    'y': sampled_seq_spatial_all[idx, 0, 1].item(),
                                    'depth': 0,
                                    'catalog_id': i
                                }])
                                gen_df = pd.concat([gen_df, new_row], ignore_index=True) 
                        

                        # modify the batch to include the generated events
                        batch_list = list(batch)
                        # Add to the end of the batch
                        for i in range(len(batch_list)):
                            batch_list[i] = torch.cat((batch_list[i], gen_events[:, :, i]), dim=1)
                            # remove the first element of the batch
                            batch_list[i] = batch_list[i][:, 1:]
                            # remove which over the end time
                            batch_list[i] = batch_list[i][which_under_end_time,:]

                        # Convert the list back to a tuple 
                        batch = tuple(batch_list)


                    # perform azimuthal equidistant projection inverse on x and y  
                    gen_df['lat'], gen_df['lon'] = azimuthal_equidistant_inverse(gen_df['x'], gen_df['y'], center_lat, center_lon)

                    # only keep lat lon mag','time_string',,'depth','catalog_id
                    gen_df = gen_df[['lon','lat','mag','time_string','depth','catalog_id']]

                    # sort the df by catalog_id then time_string
                    gen_df = gen_df.sort_values(by=['catalog_id','time_string'])

                    path_to_forecasts = './'
                    # path_to_forecasts = '/user/work/ss15859/'

                    # write batch to csv
                    if not os.path.exists(path_to_forecasts +'SMASH_daily_forecasts'):
                        os.mkdir(path_to_forecasts +'SMASH_daily_forecasts')

                    if not os.path.exists(path_to_forecasts +'SMASH_daily_forecasts/{}'.format(opt.dataset)):
                        os.mkdir(path_to_forecasts +'SMASH_daily_forecasts/{}'.format(opt.dataset))
                    
                    if not os.path.exists(path_to_forecasts +'SMASH_daily_forecasts/{}/CSEP_day_{}.csv'.format(opt.dataset, opt.day_number)):
                        gen_df.to_csv(path_to_forecasts +'SMASH_daily_forecasts/{}/CSEP_day_{}.csv'.format(opt.dataset, opt.day_number), index=False)
                    else:
                        gen_df.to_csv(path_to_forecasts +'SMASH_daily_forecasts/{}/CSEP_day_{}.csv'.format(opt.dataset, opt.day_number), mode='a', header=False, index=False)
                    


                if opt.mode == 'sample':
                    print('Sampling Done!')
                    break

  
            if opt.mode == 'train':
                torch.save(Model.state_dict(), model_path+'model_{}.pkl'.format(itr))
                print('Model Saved to {}'.format(model_path+'model_{}.pkl').format(itr))
            
                            
        
            # Validation set evaluation
            loss_test_all = 0.0
            total_num = 0
            

            for batch in valloader:
                event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)
                loss = Model.decoder(torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1), enc_out_non_mask)
                
                loss_test_all += loss.item() * event_time_non_mask.shape[0]
                total_num += event_time_non_mask.shape[0]

            # Compute the average validation loss
            avg_loss_test = loss_test_all / total_num if total_num > 0 else float('inf')

            # Check if this is the best model so far
            if avg_loss_test < min_loss_test:
                print('---------------------------- Model Updated! ------------------------------------')
                torch.save(Model.state_dict(), opt.save_path + 'model_best.pkl')
                min_loss_test = avg_loss_test  # Update best loss
                early_stop = 0  # Reset early stopping counter
            else:
                early_stop += 1  # Increase early stopping counter

            # Early stopping condition
            if early_stop >= 15 and opt.mode == 'train':
                print("Early stopping triggered after 150 epochs without improvement.")
                break


       
        if itr < warmup_steps:
            for param_group in optimizer.param_groups:
                lr = LR_warmup(1e-3, warmup_steps, itr)
                param_group["lr"] = lr
        else:
            for param_group in optimizer.param_groups:
                lr = 1e-3- (1e-3 - 5e-5)*(itr-warmup_steps)/opt.total_epochs
                param_group["lr"] = lr


        Model.train()
        loss_all, total_num = 0.0, 0.0
        for batch in trainloader:

            event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)
            loss = Model.decoder(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1),enc_out_non_mask)

            optimizer.zero_grad()
            loss.backward()
            loss_all += loss.item() * event_time_non_mask.shape[0]


            torch.nn.utils.clip_grad_norm_(Model.parameters(), 1.)
            optimizer.step() 
            
            step += 1

            total_num += event_time_non_mask.shape[0]
        
        if device.type == 'cuda':
            with torch.cuda.device("cuda:{}".format(opt.cuda_id)):
                torch.cuda.empty_cache()

        print('------- Training ---- Epoch: {} ;  Loss: {} --------'.format(itr, loss_all/total_num))




