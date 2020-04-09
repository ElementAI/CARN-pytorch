import os
import argparse
import pandas as pd
import pprint
import torch 

import exp_configs
from src.backbones import get_backbone
from src import datasets
from src import models

from haven import haven_utils as hu
from haven import haven_chk as hc
from haven import haven_jobs as hj


def trainval(exp_dict, savedir_base, n_workers, test_only, reset=False):
    # bookkeeping
    # ---------------

    # get experiment directory
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)

    if reset:
        # delete and backup experiment
        hc.delete_experiment(savedir, backup_flag=True)
    
    # create folder and save the experiment dictionary
    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, 'exp_dict.json'), exp_dict)
    pprint.pprint(exp_dict)
    print('Experiment saved in %s' % savedir)

    # Dataset
    # -----------

    # train loader
    train_loader = datasets.get_loader("train", exp_dict, n_workers, test_only=test_only)

    # val loader
    val_loader = datasets.get_loader("test", exp_dict, n_workers, test_only=test_only)

    # Model
    # -----------
    model = models.get_model(exp_dict)

    # Checkpoint
    # -----------
    model_path = os.path.join(savedir, 'model.pth')
    score_list_path = os.path.join(savedir, 'score_list.pkl')

    if os.path.exists(score_list_path):
        # resume experiment
        model.set_state_dict(hu.torch_load(model_path))
        score_list = hu.load_pkl(score_list_path)
        s_epoch = score_list[-1]['epoch'] + 1
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # Train & Val
    # ------------
    print('Starting experiment at epoch %d' % (s_epoch))

    for e in range(s_epoch, 10):
        score_dict = {}

        # Train the model
        score_dict.update(model.train_on_loader(train_loader))

        # Validate the model
        score_dict.update(model.test_on_loader(val_loader))

        # Get metrics
        # score_dict['train_loss'] = train_dict['train_loss']
        # score_dict['val_acc'] = val_dict['val_acc']
        score_dict['epoch'] = e

        # Add to score_list and save checkpoint
        score_list += [score_dict]

        # Report & Save
        score_df = pd.DataFrame(score_list)
        print(score_df.tail())
        hu.torch_save(model_path, model.get_state_dict())
        hu.save_pkl(score_list_path, score_list)
        print('Checkpoint Saved: %s' % savedir)

    print('experiment completed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-nw', '--n_workers', type=int, default=0)
    parser.add_argument('-t', '--test_only', type=int, default=0)
    parser.add_argument("-r", "--reset",  default=0, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-j", "--run_jobs", type=int, default=0)

    args = parser.parse_args()

    # Collect experiments
    # -------------------
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, "exp_dict.json"))        
        
        exp_list = [exp_dict]
        
    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]

    # launch jobs
    if args.run_jobs:
            # launch jobs
            from haven import haven_jobs as hjb
            run_command = ('python trainval.py -ei <exp_id> -sb %s -d %s -nw 1' %  (args.savedir_base, args.datadir_base))
            job_config = {'volume': "",
                        'image': "",
                        'bid': '1',
                        'restartable': '1',
                        'gpu': '4',
                        'mem': '30',
                        'cpu': '2'}
            workdir = os.path.dirname(os.path.realpath(__file__))

            hjb.run_exp_list_jobs(exp_list, 
                                savedir_base=args.savedir_base, 
                                workdir=workdir,
                                run_command=run_command,
                                job_config=job_config)
    else:
        # run experiments
        for exp_dict in exp_list:
            # do trainval
            trainval(exp_dict=exp_dict,
                    savedir_base=args.savedir_base,
                    n_workers=args.n_workers,
                    test_only=args.test_only,
                    reset=args.reset)