import hashlib
import json
import logging
import os
import random
import re
import shutil
import copy

import torch
from torch.backends import cudnn
from multi_train_utils.distributed_utils import log_to_console
import numpy as np
from numpy import prod


def setup_seed(args, params):
    seed = args.seed
    torch.manual_seed(seed)
    if params.cuda_seed:
        log_to_console('Set one cuda seed for all gpus! Probably degrades performance')
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed(seed + args.rank)

    np.random.seed(seed)
    random.seed(seed)

    if params.cudnn_enabled:

        cudnn.deterministic = False
        cudnn.benchmark = params.cudnn_benchmark
        if cudnn.benchmark:
            log_to_console("cudnn.deterministic: ", cudnn.deterministic)
            log_to_console('Cudnn benchmark has been turned on! Check out the performance difference')
    else:
        cudnn.enabled = False
        log_to_console("Cudnn backends has been turned off, probably slows down speed!")


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def copy(self):
        return copy.deepcopy(self)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves mymodel and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains mymodel's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best mymodel seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads mymodel parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) mymodel for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def search_experiment(id: int, name: str, experiments_root=r'../experiments'):
    assert id is not None and name is not None

    all_dirs = os.listdir(experiments_root)
    found_dir = []

    for exp in all_dirs:
        exp_path = os.path.join(experiments_root, exp)
        if os.path.isfile(exp_path):
            continue 

        nums = re.findall(r'\d+', exp)
        if len(nums) > 0:
            if int(nums[0]) == id:
                found_dir.append(exp)

    if len(found_dir) > 1:  
        print(found_dir)
        raise ValueError('More than one experiment found by id! Rename them!')

    elif len(found_dir) == 0:  
        experiment_dir = f'exp{id}_{name}'.format(id=id, name=name)
        if os.path.isdir(experiment_dir):
            raise ValueError('Already exists a dir')
        experiment_dir_path = os.path.join(experiments_root, experiment_dir)
        print('Making dir for new experiment: ', experiment_dir)
        os.mkdir(experiment_dir_path)
        return experiment_dir_path

    else:
        found_dir = found_dir[0]
        found_exp_name = found_dir.replace(f'exp{id}_'.format(id=id), '')  
        if found_exp_name != name:
            raise ValueError('Experiment name does not match, expect {}'.format(found_exp_name))
        found_exp_path = os.path.join(experiments_root, found_dir)
        return found_exp_path


def find_all_experiment_dirs(parent_dir_path, experiments_list=None, find_file=None, exclude_file=None):
    '''
    找到parent dir 下存在指定格式文件的所有目录
    :param parent_dir_path:
    :param experiments_list:
    :param find_file:
    :return:
    '''

    if not exclude_file:
        exclude_file = ['metrics_train_set.json', 'metrics_test_set.json'] 
        exclude_file = [exclude_file]
    check_exclude_file = lambda dir_path: prod([os.path.isfile(os.path.join(dir_path, file)) for file in exclude_file])

    if not find_file:
        find_file = 'checkpoint.pth'
    if isinstance(find_file, list):
        check_exists_file = lambda dir_path: prod([os.path.isfile(os.path.join(dir_path, file)) for file in find_file])
    else:
        check_exists_file = lambda dir_path: os.path.isfile(os.path.join(dir_path, find_file))

    check_experiment_dir = lambda dir_path: check_exists_file(dir_path) and not check_exclude_file(
        dir_path)  

    assert experiments_list is not None, 'Experiment list is none!'

    if check_experiment_dir(parent_dir_path):
        if parent_dir_path not in experiments_list:
            experiments_list.append(parent_dir_path)
            print('Append ', parent_dir_path)

    # 遍历子级目录
    for root, dirs, files in os.walk(parent_dir_path, False):
        for child_dir in dirs:
            child_dir_path = os.path.join(root, child_dir)
            if check_experiment_dir(child_dir_path):
                if child_dir_path not in experiments_list:
                    experiments_list.append(child_dir_path)
                    # print('Append ', child_dir_path)
        pass


def backup_code(target_experiment_dir):

    code_backup_dir = os.path.join(target_experiment_dir, 'code_backup')
    if not os.path.isdir(code_backup_dir):
        os.mkdir(code_backup_dir)  
    print('Backup codes at :', code_backup_dir)
    for root, code_dirs, files in os.walk('./'):  
        if '__pycache__' in root or '.@__thumb' in root or 'experiments' in root or 'configs' in root:
            continue
        for code_dir in code_dirs:
            if code_dir == '.@__thumb' or code_dir == '__pycache__' or code_dir == 'experiments' :  
                continue
            sub_backup_dir = os.path.join(code_backup_dir, root.replace('./', ''), code_dir)
            if not os.path.isdir(sub_backup_dir):
                # print('Making code sub dir: ', sub_backup_dir)
                os.mkdir(sub_backup_dir)

        for file in files:
            if file.endswith('.py'):
                backup_file_path = os.path.join(code_backup_dir, root.replace('./', ''), file)
                src_file_path = os.path.join(root, file)
                # print(src_file_path)
                shutil.copy(src_file_path, backup_file_path)


def set_params_default(default_params: dict, parent_dir='./experiments'):
    experiments_list = []
    find_all_experiment_dirs(parent_dir_path=parent_dir, experiments_list=experiments_list, find_file='params.json',
                             exclude_file='nothing')
    for exp_path in experiments_list:
        params_path = exp_path + os.sep + 'params.json'
        params = Params(params_path)
        for key in default_params.keys():
            params.dict.setdefault(key, default_params[key])
            print(exp_path, key, params.dict[key])
        params.save(params_path)


def rename_param_name(old_key_name: str, new_key_name: str):
    experiments_list = []
    find_all_experiment_dirs(parent_dir_path='./experiments', experiments_list=experiments_list,
                             find_file='params.json', exclude_file='nothing')
    for exp_path in experiments_list:
        params_path = exp_path + os.sep + 'params.json'
        params = Params(params_path)
        value = params.dict.pop(old_key_name)
        params.dict.update({new_key_name: value})
        print(exp_path, new_key_name, params.dict[new_key_name])
        params.save(params_path)


def delete_param(abandoned_key: str):
    experiments_list = []
    find_all_experiment_dirs(parent_dir_path='./experiments', experiments_list=experiments_list,
                             find_file='params.json', exclude_file='nothing')
    for exp_path in experiments_list:
        params_path = exp_path + os.sep + 'params.json'
        params = Params(params_path)
        value = params.dict.pop(abandoned_key)
        print(exp_path, ' delete {}:{}'.format(abandoned_key, value))
        params.save(params_path)


def get_hash_str(string:str):
    return hashlib.md5(string.encode('utf-8')).hexdigest()


if __name__ == '__main__':
    pass
    default_params = {
        # 'weight_decay':0,
        # 'batch_norm_to_instance_norm':False
        # "normalize_R":True,
        # 'freeze_layers':False
        # "pretrained":False,
        # "cudnn_benchmark":False,
        # "optimizer":"Adam",
        # "scheduler":"step_lr_small_step",
        # "warmup_lr":False,
        # 'cuda_seed':False,
        # "transform_type":"normalize"
        # "shuffle_dataset":False
        # "cudnn_enabled":True
        # "spectrum_sample":11
        # "image_format": ".exr",
        # "use_white_background":True
        # "model":"UNet",
        # "loss":"Image2MSE",
        # "no_clip":False,
        # "act_func":"sigmoid",
        "criternion":"masked_psnr"
    }
    set_params_default(default_params=default_params)


    """
    experiment_dirs = []
    parent_dir = r'./experiments/exp25_model_with_best_weight_decay/model'
    find_all_experiment_dirs(parent_dir_path=parent_dir, experiments_list=experiment_dirs,exclude_file='loss_result.json')
    for exp in experiment_dirs:
        checkpoint = torch.load(exp+os.sep+'checkpoint.pth', map_location='cpu')  # 读取checkpoint文件

        train_loss = checkpoint.get('latest_train_loss')
        if not train_loss:
            train_loss = -1
        min_test_loss = checkpoint['min_test_loss']

        with open(exp + os.sep + 'loss_result.json', 'w') as file:
            loss_result = {
                "latest_train_loss": train_loss,  # 保存训练到最后的loss情况，方便调参
                'min_test_loss': min_test_loss,
            }
            print('Saving loss result at ',exp)
            json.dump(loss_result, file, indent=4)

    from synthesize_results import synthesize_loss_to_md
    loss_info = synthesize_loss_to_md(parent_dir)    """
