from sklearn.utils import shuffle
import pandas as pd
import argparse
import time
import sys
import os
'''
STEP 4
USER LABELING

This code is to label users by configuration.

          read configration <---------------------------------------<
                 v                                                  |
          for chunk in data                                         |
                 v                                                  |
            label data                                              |
                 |                                                  |
                 |-<if labelled data not achieve expect percentage>-^
                 |
                 |---------------------------------------------v
                 |                                             |
                 v                                             |
insert to original data as a new column                        |
                 v                                             v
     save to local in .csv & .h5           save labelled rows to local in .csv & .h5

--------------------------------------------------------

CONFIGURATION FILE(.txt)

Copy paste this text to a .txt file as configration file.
Line starts with "#" is comment, will skipped.

1. There are 2 conditions:
CONDITION 1 - Label base on single column
col1-condition

CONDITION 2 - Label base on multiple columns, logic AND
col1-condition1&COL2-condition2&... Label

CONDITION 3 - Label base on multiple columns, logic OR
col_1|col_2|col_3|...-condition Label

2. How to write condition:
a. If 50 <= x <= 100
   x-[50,100]

b. If 50 < x < 100
   x-[51,99]

c. If x == ':)'
   x-[':)']

d. If x != ':)'
   x-[!':)']

3. Example:
col1-[-10:50] typeA
col1-[50,100]&col3-[':)'] typeB
col2-[':('] typeC
col3-[!':)'] typeD
col4-[0:100] typeE
col2|col3-[':3'] typeF

'''

class Labeling:
    def __init__(self, configs, fi, dir_o, ratio, chunksize):
        self.configs = configs
        self.fi = fi
        self.dir_o = dir_o
        self.ratio = float(ratio)
        self.chunksize = chunksize

        if not os.path.exists(self.dir_o):
            os.mkdir(self.dir_o)
    '''
    k: condition_key
    i: row_idx
    v: label_value
    '''
    # Match text-based conditions
    def match_text(self, df, k, i):

        target = k.split('-')[0]
        condition = k.split('-')[1]
        
        if '!' in condition:
            condition = condition.lstrip("[!'").rstrip("']")
            # if df.loc[i, target] != condition:
            if condition != df.loc[i, target]:
                return True
        else:
            condition = condition.lstrip("['").rstrip("']")
            # if df.loc[i, target] == condition:
            if condition in df.loc[i, target]:
                return True
        
        return False

    # match number
    def match_num(self, df, k, i):

        if '-' not in k or len(k.split('-')) < 2:
            raise ValueError(f"Invalid format for key: {k}")

        target = k.split('-')[0]
        condition = k.split('-')[1]

        # Validate condition format
        if ':' not in condition or not condition.startswith('[') or not condition.endswith(']'):
            raise ValueError(f"Invalid condition format: {condition}")

        condition_1 = float(condition.split(':')[0].lstrip("["))
        condition_2 = float(condition.split(':')[1].rstrip("]"))

        if condition_1 <= float(df.loc[i, target]) <= condition_2:
            return True
        
        return False
    
    def label_data(self, df, i):
        label = ''

        for k, v in self.configs.items():
            # handle single condition (no '&' or '|')
            if '&' not in k and '|' not in k:
                # match text
                if "'" in k:
                    if self.match_text(df, k, i):
                        if label != '':
                            label += f'.{v}'
                        else:
                            label = v
                
                # match number
                else:
                    if self.match_num(df, k, i):
                        if label != '':
                            if v not in label:
                                label += f'.{v}'
                        else:
                            label = v
            
            # handle multiple condition, logic AND
            elif '&' in k:
                matched = True
                matched_labels = []

                for sub_condition in k.split('&'):
                    # match text
                    if "'" in sub_condition:
                        if not self.match_text(df, sub_condition, i):
                            matched = False
                            break

                    # match number
                    else:
                        if not self.match_num(df, sub_condition, i):
                            matched = False
                            break

                    matched_labels.append(v)

                if matched:
                    if label != '':
                        for matched_label in matched_labels:
                            if matched_label not in label:
                                label += f'.{matched_label}'
                    else:
                        for matched_label in matched_labels:
                            label += f'.{matched_label}'

                        label = label.lstrip('.')

            # with multiple condition, logic OR
            elif '|' in k:
                matched_labels = []

                # col_1|col_2|col_3-['xxx'] L1
                # col_1|col_2|col_3-['xxx'] L2
                #             |
                #             V
                # col_1: [L1, L2]
                # col_2: [L1, L2]
                # col_3: [L1, L2]

                cols = k.split('-')[0]
                condition = k.split('-')[1].split(' ')[0]
                condition = condition.lstrip("['").rstrip("']")

                for col in cols.split('|'):
                    # if df.loc[i, elm] == condition:
                    if condition in df.loc[i, col]:
                        matched_labels.append(v)

                if matched_labels:
                    if label != '':
                        for matched_label in matched_labels:
                            if matched_label not in label:
                                label += f'.{matched_label}'

                    else:
                        for matched_label in matched_labels:
                            label += f'.{matched_label}'

                        label = label.lstrip('.')

        # 去重逻辑
        if label:
            unique_labels = list(set(label.split('.')))
            label = '.'.join(sorted(unique_labels))

        return label


    def save2csv(self, df, fo_name):
        fo = os.path.join(self.dir_o, fo_name)
        print(f'[*] Saving data to {fo}')
        df.to_csv(fo, index=False)
        print('[*] Done.')
    
    def save2hdf5(self, df, fo_name):
        fo = os.path.join(self.dir_o, fo_name)
        print(f'[*] Saving data to {fo}')
        df.to_hdf(fo, key='df', mode='w')
        print('[*] Done.')

    def preview(self, df):
        print(f'[*] Data preview:')
        print('------------------------------------------------------------------------')
        print(df.info())

        print('\n')
        if df.shape[0] < 5:
            for i in df.index:
                print(f'{df.iloc[i]}\n')
        
        else:
            for i in range(5):
                print(f'{df.iloc[i]}\n')
    
        print('------------------------------------------------------------------------')

    def run(self):
        start = time.time()

        print(f'[*] Reading from {os.path.basename(self.fi)}...')
        df = pd.read_csv(self.fi)
        df = shuffle(df)
        df.reset_index(drop=True, inplace=True)
        self.preview(df)
        print('[*] Done.')

        print('[*] Starting labeling data...')
        chunk_iterator = [df.iloc[i:(i + self.chunksize)] for i in range(0, len(df), self.chunksize)]

        df_len = df.shape[0]
        total_labels = round(df_len*self.ratio, 0)

        if total_labels == 0:
            total_labels = 1

        labelled_rows = []
        label_col = []
        for chunk_df in chunk_iterator:
            print(f'\r[*] Progress: {round(round(len(labelled_rows)/total_labels, 3)*100 , 2)}%', end='')
            for i in chunk_df.index:
                if len(labelled_rows) == total_labels:
                    break

                else:
                    label = self.label_data(chunk_df, i)
                    if label != '':
                        labelled_rows.append(chunk_df.loc[i])
                        label_col.append(label.replace('\n', '').replace('"', ''))
                    else:
                        label_col.append('/')
        
        print(f'\r[*] Progress: 100%       ', end='')
        print('\n[*] Done.')

        # Complete label column
        for i in range(df_len - len(label_col)):
            label_col.append(pd.NA)

        # Insert label column to df
        df.insert(df.shape[1], 'labels', label_col, allow_duplicates=True)

        # save to local
        self.save2csv(df, os.path.basename(self.fi).replace('.csv', '_labeled.csv'))

        self.preview(df)

        # save labeled rows to csv
        print('[*] Saving labeled rows...')
        labelled_df = df[df['labels'] != '/']
        
        # save to local
        # self.save2csv(labelled_df, os.path.basename(self.fi).replace('.csv', '_labeled_only.csv'))

        time_spend = time.time() - start
        if time_spend < 60:
            print(f'\n Time spend: {round(time_spend, 2)} second.\n')
        else:
            print(f'\n Time spend: {round((time_spend)/60, 2)} minutes.\n')

# Read configration
def read_config(config_fi):
    configs = {}
    with open(config_fi) as filo:
        for line in filo:
            if not line.startswith('#') and line != '\n':
                contents = line.replace('\n', '').split(' ')

                configs.setdefault(contents[0], '')
                configs[contents[0]] = contents[1]

    # check config content
    print(f'[*] Config Content:')
    for k, v in configs.items():
        print(f'\t{k}: {v}')

    return configs

# For ratio option
def restrict_float(n):
    try:
        if float(n) == 0:
            raise argparse.ArgumentTypeError(f'[*] {n} can not be 0, need in range of 0 - 1(include)!')
        elif float(n) < 0 or float(n) > 1:
            raise argparse.ArgumentTypeError(f'[*] {n} not in range of 0 - 1(include)!')
        else:
            return n
    except ValueError:
        raise argparse.ArgumentTypeError(f'[*] {n} not in range of 0 - 1(include)!')


if __name__ == '__main__':
    help = \
'''
Example:
CONDITION 1
    statistic.csv
    python user_labeling.py -CONFIG config.txt -I statistic.csv -O path/to/directory -S 10000
    [*] -S is optional, default is 10000. This means the chunk size processed each time, depend on your RAM.

CONDITION 2
    path/to/my_statistics/
    python user_labeling.py -CONFIG config.txt -I path/to/my_statistics -O path/to/directory -S 10000 -R T
    [*] To process multiple files in a row, use -R T to achieve. This will preprocess all files within the path/to/my_statistics/ directory.


------------------------------------COPY BELOW------------------------------------
# CONFIGURATION FILE(.txt)
# 
# Copy paste this text to a .txt file as configration file.
# Line starts with "#" is comment, will skipped.
# 
# 1. There are 2 conditions:
# CONDITION 1 - Label base on single column
# col1-condition
# 
# CONDITION 2 - Label base on multiple columns, logic AND
# col1-condition1&COL2-condition2&... Label
# 
# CONDITION 3 - Label base on multiple columns, logic OR
# col_1|col_2|col_3|...-condition Label
# 
# 2. How to write condition:
# a. If 50 <= x <= 100
#    x-[50,100]
# 
# b. If 50 < x < 100
#    x-[51,99]
# 
# c. If x == ':)'
#    x-[':)']
# 
# d. If x != ':)'
#    x-[!':)']
# 
# 3. Example:
# col1-[-10:50] typeA
# col1-[50,100]&col3-[':)'] typeB
# col2-[':('] typeC
# col3-[!':)'] typeD
# col4-[0:100] typeE
# col2|col3-[':3'] typeF
------------------------------------COPY ABOVE------------------------------------
'''

    parser = argparse.ArgumentParser(description='DATA AUTO LABELING',
                                     epilog=help,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-CONFIG', '--CONFIGRATION', required=True, help='For content of configration file, use -h to check')
    parser.add_argument('-I', '--INPUT_FILE', required=True)
    parser.add_argument('-O', '--OUTPUT_DIRECTORY', required=True, help='File name is not required, just need path.')
    parser.add_argument('-RATIO', '--LABEL_RATIO', required=True, type=restrict_float, help='How much data your want to label(0-1)? E.g. 0.5')
    parser.add_argument('-S', '--CHUNK_SIZE', default=10000, help='Set chunk size for processing data chunk by chunk, default is 10000.')
    parser.add_argument('-R', '--RECURSIVE', default=False, choices=['T', 'F'], help='Optional, choose from T&F. This is for process multiple files in a row. If T, -I should be a directory path.')

    args = parser.parse_args()
    config_fi = args.CONFIGRATION
    fi = args.INPUT_FILE
    dir_o = args.OUTPUT_DIRECTORY
    ratio = args.LABEL_RATIO
    chunksize = args.CHUNK_SIZE
    recursive = args.RECURSIVE

    if recursive == 'T':
        recursive = True
    else:
        recursive = False
    
    # For processing files in a row
    if recursive:
        if not os.path.isdir(fi):
            sys.exit(f'[*] Error: you have use "-R T", but {dir_o} is not a directory!')
        
        if os.path.isdir(dir_o):
            sys.exit(f'[*] Error: {dir_o} need to be a directory!')
        
        # Praise the God of all Machines - The mighty Omnissiah!
        # prayer().toPray()

        # Read configration
        print('[*] Reading configration file...')
        configs = read_config(config_fi)
        print('[*] Done.')
        
        files = os.listdir(fi)
        i = 1
        for file in files:
            if file != '.DS_Store':
                curr_fi = os.path.join(fi, file)
                print(f'[*] Current file - {file}, {len(files) - i} remaining.')

                # MAIN
                labeling = Labeling(configs, curr_fi, dir_o, ratio, chunksize)
                labeling.run()
            
            i += 1

    # For processing single file
    else:
        if not os.path.exists(fi):
            sys.exit(f'[*] Error: {fi} not exist!')
        
        if not os.path.isdir(dir_o):
            sys.exit(f'[*] Error: {dir_o} need to be a directory!')

        # Praise the God of all Machines - The mighty Omnissiah!
        # prayer().toPray()

        # Read configration
        print('[*] Reading configration file...')
        configs = read_config(config_fi)
        print('[*] Done.')

        # MAIN
        labeling = Labeling(configs, fi, dir_o, ratio, chunksize)
        labeling.run()
