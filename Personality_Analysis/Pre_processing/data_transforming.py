import multiprocessing as mp
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import argparse
import time
import sys
import os

'''
STEP 2
DATA TRANSFORMATION

Transformation will be performed around user, the code will split event_time
column. And filter records depend on user's purchase frequency

COLUMN SEQUENCE
(event_time -> event_date,event_time)

    for file in all_files <-----------------------<
           v                                      |
    for chunk in data <---------<                 |
           v                    |                 |
  split event_time column       |                 |
           v                    |                 |
     record user id             |                 |
           |                    |                 |
           |-<if loop not over>-^                 |
           |                                      |
           v                                      |
     concat chunks                                |
           v                                      |
  drop event_time column                          |
           v                                      | 
insert new event_time columns                     |
           v                                      | 
sort column(COLUMN SEQUENCE) -<if loop not over>--^
           v
     concat files
           v
save to local in .csv & .h5
           |
<Split data base on user id>
           |
           v
create a new directory - user_records
           v
 for each user in records <---------------------<
           v                                    |
save to local in .csv & .h5 -<if loop not over>-^
'''

class transformation:
    def __init__(self, fi, dir_o, threshold, chunksize, rebuild, multiProcessing, dfs):
        self.fi = fi
        self.dir_o = dir_o
        self.threshold = threshold
        self.chunksize = chunksize
        self.rebuild = rebuild
        self.multiProcessing = multiProcessing

        self.dfs = dfs
        self.usr_lst = []

        if not os.path.exists(self.dir_o):
            os.mkdir(self.dir_o)
    
    def save2csv(self, df, dir_o, fo_name):
        fo = os.path.join(dir_o, fo_name)
        df.to_csv(fo, index=False)
    
    def save2hdf5(self, df, dir_o, fo_name):
        fo = os.path.join(dir_o, fo_name)
        df.to_hdf(fo, key='df', mode='w')

    def preview(self, df):
        print(f'[*] Data preview:')
        print('------------------------------------------------------------------------')
        print(df.info())

        print('\n')
        for i in range(5):
            print(f'{df.iloc[i]}\n')
    
        print('------------------------------------------------------------------------')
    
    def rebulid_time(self, dfObj, new_time):
        # old format
        # year-mouth-day hour:minutes:second time_zone
        Ts = dfObj.split(' ')

        new_time['date'].append(Ts[0])
        new_time['time'].append(Ts[1])
        new_time['time_zone'].append(Ts[2])

        # Convert the time to float
        time_obj = datetime.strptime(Ts[1], '%H:%M:%S')
        time_float = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
        new_time['time_float'].append(time_float)

        return new_time
    
    def MP_rebuild(self, chunk_df, new_time, PID, return_dict):
        if PID == 0:
            for i in tqdm(range(chunk_df.shape[0])):
                # record user_id
                curr_usr = chunk_df.loc[i, 'user_id']
                if curr_usr not in self.usr_lst:
                    self.usr_lst.append(curr_usr)

                # split event_time column
                new_time = self.rebulid_time(chunk_df.loc[i, 'event_time'], new_time)
        
        else:
            for i in chunk_df.index:
                # record user_id
                curr_usr = chunk_df.loc[i, 'user_id']
                if curr_usr not in self.usr_lst:
                    self.usr_lst.append(curr_usr)

                # split event_time column
                new_time = self.rebulid_time(chunk_df.loc[i, 'event_time'], new_time)

        return_dict[PID] = (new_time, self.usr_lst)



    def check_directory(self, dir):
        if not os.path.exists(dir):
            print(f'[*] Creating output directory - {os.path.basename(dir)}...')
            os.mkdir(dir)
            print('[*] Done.')
    
    def MP_generate_usr_records(self, PID, df, usr_lst, usr_start, usr_end, usr_num, usr_dir_csv, usr_dir_hdf):
        if PID == 0:
            for i in tqdm(range(usr_start, usr_end)):
                try:            
                    usr_df = df[df['user_id'] == usr_lst[i]]

                except IndexError:
                    pass
                
                else:

                    if usr_df['category_code'].nunique() > 3 and usr_df.shape[0] > int(self.threshold):
                        # save as csv file
                        self.save2csv(usr_df, usr_dir_csv, f'{usr_lst[i]}.csv')

                        # # save as HDF5 file
                        # try:
                        #     self.save2hdf5(usr_df, usr_dir_hdf, f'{usr_lst[i]}.h5')
                        # except Exception as e:
                        #     print('\n------------------------------------------------------------------------')
                        #     print(f'[*] Error: {e}')
                        #     print('[*] Skip saving as HDF5 file.')
                        #     print('------------------------------------------------------------------------')
                        #     print('')
                        #
        else:
            for i in range(usr_start, usr_end):
                try:            
                    usr_df = df[df['user_id'] == usr_lst[i]]

                except IndexError:
                    pass
                
                else:
                    # drop the users whose types of categories are less than 3


                    if usr_df['category_code'].nunique() > 3 and usr_df.shape[0] > int(self.threshold):
                        # save as csv file
                        self.save2csv(usr_df, usr_dir_csv, f'{usr_lst[i]}.csv')

                        # # save as HDF5 file
                        # try:
                        #     self.save2hdf5(usr_df, usr_dir_hdf, f'{usr_lst[i]}.h5')
                        # except Exception as e:
                        #     print('\n------------------------------------------------------------------------')
                        #     print(f'[*] Error: {e}')
                        #     print('[*] Skip saving as HDF5 file.')
                        #     print('------------------------------------------------------------------------')
                        #     print('')

    # for each user, we will generate their own records file
    def generate_usr_records(self, df, usr_lst):
        # Create directory user_records in dir_o
        usr_dir_csv = os.path.join(self.dir_o, 'user_records_csv')
        usr_dir_hdf = os.path.join(self.dir_o, 'user_records_hdf')

        self.check_directory(usr_dir_csv)
        self.check_directory(usr_dir_hdf)

        print('[*] Generating user records...')

        if self.multiProcessing > 1:
            usr_num = round(len(usr_lst)/self.multiProcessing) + 1
            usr_start = 0
            usr_end = usr_start + usr_num
            processings = []
            speed_start = time.time()

            for i in range(self.multiProcessing):
                processing = mp.Process(
                    target=self.MP_generate_usr_records,
                    args=(i, df, usr_lst, usr_start, usr_end, usr_num, usr_dir_csv, usr_dir_hdf)
                )
                processing.start()

                if i == 0:
                    print(f'[*] Launch PROCESSING-{i}')
                else:
                    print(f'\n[*] Launch PROCESSING-{i}')

                processings.append(processing)

                usr_start = usr_end
                usr_end += usr_num
            
            for j in range(len(processings)):
                processings[j].join()

                if j == 0:
                    print(f'\n[*] PROCESSING-{j} done.')
                else:
                    print(f'[*] PROCESSING-{j} done.')
        
        else:
            speed_start = time.time()
            usr_num = len(usr_lst)

            for i in tqdm(range(usr_num)):
                #print(f'\r[*] Progress: {round(round(i/usr_num, 3)*100 , 2)}% [{i+1}/{usr_num}]  ', end='')
                usr_df = df[df['user_id'] == usr_lst[i]]

                if 'time_float' not in usr_df.columns:
                    print(f"Warning: 'time_float' column not found in data for user {usr_lst[i]}")
                    continue

                if usr_df['category_code'].nunique() > 3 and usr_df.shape[0] > int(self.threshold):
                    # save as csv file
                    self.save2csv(usr_df, usr_dir_csv, f'{usr_lst[i]}.csv')

                    # # save as HDF5 file
                    # try:
                    #     self.save2hdf5(usr_df, usr_dir_hdf, f'{usr_lst[i]}.h5')
                    # except Exception as e:
                    #     print('\n------------------------------------------------------------------------')
                    #     print(f'[*] Error: {e}')
                    #     print('[*] Skip saving as HDF5 file.')
                    #     print('------------------------------------------------------------------------')
                    #     print('')
        
        speed_time = time.time() - speed_start
        print(f'\n[*] Done, processing speed: {round(len(usr_lst)/speed_time)} rows/s.')

    def MP_usr_lst_prepare(self, PID, chunk_df, return_dict):
        usr_lst = []
        if PID == 0:
            for i in tqdm(range(chunk_df.shape[0])):
                # record user_id
                curr_usr = chunk_df.loc[i, 'user_id']
                if curr_usr not in usr_lst:
                    usr_lst.append(curr_usr)
        
        else:
            for i in chunk_df.index:
                # record user_id
                curr_usr = chunk_df.loc[i, 'user_id']
                if curr_usr not in usr_lst:
                    usr_lst.append(curr_usr)
            
        return_dict[PID] = usr_lst

    def run(self):
        start = time.time()

        print(f'[*] Reading from {os.path.basename(self.fi)}...')
        df = pd.read_csv(self.fi)
        df.reset_index(drop=True, inplace=True)
        self.preview(df)

        df_len = df.shape[0]

        print(f'\n[*] Start transforming data({df_len} records)...')

        # rebuild event_time
        if rebuild:
            new_time = {'time_zone': [],
                        'time': [],
                        'date': [],
                        'time_float':[]}

            if self.multiProcessing > 1: 
                self.chunksize = round(df.shape[0]/self.multiProcessing) + 1
                chunk_iterator = [df.iloc[i:(i + self.chunksize)] for i in range(0, len(df), self.chunksize)]

                manager = mp.Manager()
                return_dict = manager.dict()

                i = 0
                processings = []
                speed_start = time.time()
                for chunk_df in chunk_iterator:
                    processing = mp.Process(target=self.MP_rebuild, args=(chunk_df, new_time, i, return_dict))
                    processing.start()

                    if i == 0:
                        print(f'[*] Launch PROCESSING-{i}')
                    else:
                        print(f'\n[*] Launch PROCESSING-{i}')
                    
                    processings.append(processing)
                    i += 1

                for j in range(len(processings)):
                    processings[j].join()
                    
                    if j == 0:
                        print(f'\n[*] PROCESSING-{j} done.')
                    else:
                        print(f'[*] PROCESSING-{j} done.')

                PIDs = sorted(return_dict)
                for PID in PIDs:
                    for k, v in return_dict[PID][0].items():
                        new_time[k] += v

                    for usr in return_dict[PID][1]:
                        self.usr_lst.append(usr)
            
            else:
                chunk_iterator = [df.iloc[i:(i + self.chunksize)] for i in range(0, len(df), self.chunksize)]

                speed_start = time.time()
                for chunk_df in chunk_iterator:
                    for i in chunk_df.index:
                        print(f'\r[*] Progress: {round(round((i+1)/df_len, 4)*100 , 3)}% [{i+1}/{df_len}]  ', end='')

                        # record user_id
                        curr_usr = chunk_df.loc[i, 'user_id']
                        if curr_usr not in self.usr_lst:
                            self.usr_lst.append(curr_usr)

                        # split event_time column
                        new_time = self.rebulid_time(chunk_df.loc[i, 'event_time'], new_time)

            del chunk_iterator # release pressure

            speed_time = time.time() - speed_start
            print(f'\n[*] Done, processing speed: {round(df_len/speed_time)} rows/s.')

            # drop event_time column
            print('[*] Dropping "event_time" column...')
            df.drop(columns='event_time', inplace=True)
            print('[*] Done.')

            # insert new event_time columns
            print('[*] Inserting new time columns...')

            for k, v in new_time.items():
                df.insert(0, k, v, allow_duplicates=True)

            print('[*] Done.')

            fo = os.path.basename(self.fi).replace('csv', 'time_formed.csv')
            fo_dir = os.path.join(self.dir_o, 'clean_data_time_formed')
            if not os.path.exists(fo_dir):
                os.mkdir(fo_dir)
            fo_abs = os.path.join(fo_dir, fo)
            
            print(f'[*] Saving to {fo_abs}...')
            df.to_csv(fo_abs, index=False)
            
            print('[*] Done.')

        else:
            if self.multiProcessing > 1:
                self.chunksize = round(df.shape[0]/self.multiProcessing) + 1
                chunk_iterator = [df.iloc[i:(i + self.chunksize)] for i in range(0, len(df), self.chunksize)]

                manager = mp.Manager()
                return_dict = manager.dict()

                i = 0
                processings = []
                speed_start = time.time()
                for chunk_df in chunk_iterator:
                    processing = mp.Process(target=self.MP_usr_lst_prepare, args=(i, chunk_df, return_dict))
                    processing.start()

                    if i == 0:
                        print(f'[*] Launch PROCESSING-{i}')
                    else:
                        print(f'\n[*] Launch PROCESSING-{i}')
                    
                    processings.append(processing)
                    i += 1

                for j in range(len(processings)):
                    processings[j].join()
                    
                    if j == 0:
                        print(f'\n[*] PROCESSING-{j} done.')
                    else:
                        print(f'[*] PROCESSING-{j} done.')

                PIDs = sorted(return_dict)
                for PID in PIDs:
                    curr_lst = return_dict[PID]
                    lst_len = len(curr_lst)
                    for i in range(lst_len):
                        print(f'\r[*] Adding PROCESSING-{PID}\'s data: {round(round((i+1)/lst_len, 4)*100 , 3)}% [{i+1}/{lst_len}]  ', end='')
                        if curr_lst[i] not in self.usr_lst:
                            self.usr_lst.append(curr_lst[i])

                    print('\n[*] Done.')
            
            else:
                chunk_iterator = [df.iloc[i:(i + self.chunksize)] for i in range(0, len(df), self.chunksize)]
                
                for chunk_df in chunk_iterator:
                    for i in chunk_df.index:
                        print(f'\r[*] Progress: {round(round((i+1)/df_len, 4)*100 , 3)}% [{i+1}/{df_len}]  ', end='')

                        # record user_id
                        curr_usr = chunk_df.loc[i, 'user_id']
                        if curr_usr not in self.usr_lst:
                            self.usr_lst.append(curr_usr)
                
                print('\n[*] Done.')

            del chunk_iterator # release pressure

        # sort column as below
        print('[*] Sorting data sequence...')
        if self.rebuild:
            new_order = ["user_id",
                         "date",
                         "time",
                         "time_float",
                         "time_zone",
                         "event_type",
                         # "product_id",
                         "category_code",
                         "brand",
                         "price"]

        else:
            new_order = ["user_id",
                         "date",
                         "time",
                         "time_zone",
                         "event_type",
                         # "product_id",
                         "category_code",
                         "brand",
                         "price"]
        
        df = df[new_order]
        print('[*] Done.')
        
        self.preview(df)

        if self.dfs != None:
            self.dfs.append(df)
        
        else:
            self.generate_usr_records(df, self.usr_lst)

        time_spend = time.time() - start
        if time_spend < 60:
            print(f'\n Time spend: {round(time_spend, 2)} second.\n')
        else:
            print(f'\n Time spend: {round((time_spend)/60, 2)} minutes.\n')
        
        if self.dfs != None:
            return self.dfs, self.usr_lst
        

if __name__ == '__main__':
    help = \
'''
Example:
CONDITION 1
    statistic.csv
    python data_transformation.py -I statistic.csv -O path/to/directory -T 10 -S 10000 -REBUILD T -MP 2
    [*] -T is optional, default is 0. This means only user with records number >= -T will be recorded.
    [*] -S is optional, default is 10000. This means the chunk size processed each time, depend on your RAM.
    [*] -MP is optional, default 1. This means enable multi processings processing with 2 processings.

CONDITION 2
    statistic.csv
    python data_transformation.py -I statistic.csv -O path/to/directory -T 10 -S 10000 -MP 1
    [*] For such .csv whose event_time is not separated(2019-10-01 00:00:00 UTC), -REBUILD need be T.
    [*] -S is optional, default is 10000. This means the chunk size processed each time, depend on your RAM.

CONDITION 3
    path/to/my_statistics/
    python data_transformation.py -I path/to/my_statistics -O path/to/directory -T 10 -S 10000 -R T -MP 1
    [*] To process multiple files in a row, use -R T to achieve. This will preprocess all files within the path/to/my_statistics/ directory.
'''

    parser = argparse.ArgumentParser(description='DATA TRANSFORMATION',
                                     epilog=help,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-I', '--INPUT_FILE', required=True)
    parser.add_argument('-O', '--OUTPUT_DIRECTORY', required=True, help='File name is not required, just need path.')
    parser.add_argument('-T', '--THRESHOLD', default=0, help='Set a threshold to filter user by their records number, default is 0.')
    parser.add_argument('-S', '--CHUNK_SIZE', default=10000, type=int, help='Set chunk size for processing data chunk by chunk, default is 10000.')
    parser.add_argument('-REBUILD', '--REBUILD_EVENT_TIME', default=False, choices=['T', 'F'], help='Optional, choose from T&F. This is just for data which event_time is not spearated. Such as: 2019-10-01 00:00:00 UTC')
    parser.add_argument('-R', '--RECURSIVE', default=False, choices=['T', 'F'], help='Optional, choose from T&F. This is for process multiple files in a row. If T, -I should be a directory path.')
    parser.add_argument('-MP', '--MULTI_PROCESSING', default=1, help='Optional, default is 1 processing. Entering the number of processing you want. If you specify processing number that larger than 1, chunk size will be assigned automaticly.')

    args = parser.parse_args()
    fi = args.INPUT_FILE
    dir_o = args.OUTPUT_DIRECTORY
    threshold = args.THRESHOLD
    chunksize = args.CHUNK_SIZE
    rebuild = args.REBUILD_EVENT_TIME
    recursive = args.RECURSIVE
    multiProcessing = int(args.MULTI_PROCESSING)

    if recursive == 'T':
        recursive = True
    else:
        recursive = False
    
    # For processing files in a row
    if recursive:
        if not os.path.isdir(fi):
            sys.exit(f'[*] Error: you have use "-R T", but {dir_o} is not a directory!')

        if rebuild == 'T':
            rebuild = True
        else:
            rebuild = False

        dfs = []
        usr_lst = []
        files = os.listdir(fi)
        i = 1
        for file in files:
            if file.endswith('csv'):
                curr_fi = os.path.join(fi, file)
                print(f'[*] Current file - {file}, {len(files) - i} remaining.')

                # MAIN
                transform = transformation(curr_fi, dir_o, threshold, chunksize, rebuild, multiProcessing, dfs)
                dfs, curr_usr_lst = transform.run()
                usr_lst += curr_usr_lst

            i += 1
        
        # Generating user records
        print('[*] concat data...')
        total_df = pd.concat(dfs, ignore_index=True)
        print('[*] Done.')

        del dfs # release pressure

        transform.generate_usr_records(total_df, usr_lst)

    # For processing single file
    else:
        if not os.path.exists(fi):
            sys.exit(f'[*] Error: {fi} not exist!')
        
        if not os.path.isdir(dir_o):
            sys.exit(f'[*] Error: {dir_o} need to be a directory!')

        if rebuild == 'T':
            rebuild = True
        else:
            rebuild = False

        # MAIN
        transform = transformation(fi, dir_o, threshold, chunksize, rebuild, multiProcessing, None)
        transform.run()
