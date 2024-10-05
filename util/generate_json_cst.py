import os
import random
from pathlib import Path

from argparse import ArgumentParser
import json

parser = ArgumentParser(description='Generate Jason Mayo dataset')

parser.add_argument('--data_dir', type=str, required=False, default=os.path.join(os.environ.get('STORE'), 'Mayo_CCST'), help='Path to the data directory')
parser.add_argument('--output_dir', type=Path, required=False, default='../src/json', help='Path to the output directory')
parser.add_argument('--name_out', type=str, required=False, default='mayo.json', help='Output file name')

parser.add_argument('--val_pid', type=str, required=False, default='L333', help='Patient ID for validation')
parser.add_argument('--test_pid', type=str, required=False, default='L310', help='Patient ID for testing')

parser.add_argument('--seed', type=int, required=False, default=42, help='Random seed')

args = parser.parse_args()

random.seed(args.seed)

def check_dir(path):
    assert os.path.isdir(path), f'Path {path} does not exist'

def check_file(path):
    assert os.path.isfile(path), f'File {path} does not exist'

def get_file_list(path, val_pid, test_pid):
    train_list = []
    val_list = []
    test_list = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.IMA'):
                abs_path = os.path.abspath(root)
                if val_pid in abs_path:
                    val_list.append(abs_path)
                elif test_pid in abs_path:
                    test_list.append(abs_path)
                else:
                    train_list.append(abs_path)
                
    return train_list, val_list, test_list

def main():
    check_dir(args.data_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_list, val_list, test_list = get_file_list(args.data_dir, args.val_pid, args.test_pid)
    random.shuffle(train_list)

    dic_jason = {}

    print('Total data length: ', len(train_list) + len(val_list) + len(test_list))

    # Training
    print('Training data length: ', len(train_list))
    list_data_file = []
    for path in train_list:
        dict_sample = {'filename': path}
        list_data_file.append(dict_sample)

    dic_jason['train'] = list_data_file

    # Validation
    print('Validation data length: ', len(val_list))
    list_data_file = []
    for path in val_list:
        dict_sample = {'filename': path}
        list_data_file.append(dict_sample)

    dic_jason['val'] = list_data_file

    # Testing
    print('Testing data length: ', len(test_list))
    list_data_file = []
    for path in test_list:
        dict_sample = {'filename': path}
        list_data_file.append(dict_sample)

    dic_jason['test'] = list_data_file

    # Save
    jason_file = open(os.path.join(args.output_dir, args.name_out), 'w')
    json.dump(dic_jason, jason_file, indent=4)
    jason_file.close()

    print('Jason file saved')

if __name__ == '__main__':
    main()
