import multiprocessing as mp
import pandas as pd
import argparse
import time
import sys
import os

'''
DATA PREPROCESSING

In optional, user can choose to separate the category_code
For example:
  category_code
  electronics.smartphone
The expected form should be:
  category_code_0 category_code_1
  electronics	    smartphone	

PIPELINE FLOWCHART
chunk in data <---------------------------< 
       V                                  | 
   drop columns                           |
       v                                  |
    drop NA -> append to list             |
                     |                    |
                     |-<if loop not over>-^
                     |
                     v
               concat chunks ---------------v (optional)
                     |                      |
                     |              for chunk in data <------------------------< 
                     |                      v                                  |
                     |             rebuild category_code                       |
                     |                      v                                  |
                     |                 drop columns                            |
                     |                      v                                  |
                     |                   drop NA -> append to list             |
                     |                                    |                    |
                     |                                    |-<if loop not over>-^
                     |                                    |
                     |                                    V
                     |                              concat chunks
                     |                                    v
                     v                                    |
          save to local in .csv & .h5 <-------------------<
                     v                                    
                 plot tree
                     v
               save to local
'''

class preprocessing:
    def __init__(self, fi, dir_o, chunksize, dropped_cols, rebuild, multiProcessing, filtering):
        self.fi = fi
        self.dir_o = dir_o
        self.chunksize = chunksize
        self.dropped_cols = dropped_cols
        self.rebuild = rebuild
        self.multiProcessing = multiProcessing
        self.filtering = filtering
        
        self.dir_o = os.path.join(self.dir_o, os.path.basename(self.fi).split('.')[0])

        if not os.path.exists(self.dir_o):
            os.mkdir(self.dir_o)

    # drop the chosen column
    def drop_col(self, df):
        # check if column name is correct or not
        for dropped_col in self.dropped_cols:
            if dropped_col not in df.columns:
                sys.exit(f'\n[*] Error: dropped column "{dropped_col}" not exist!\n[*] Available columns are: {[col for col in df.columns]}.')

        for dropped_col in self.dropped_cols:
            df.drop(dropped_col, axis=1, inplace=True)
        
        return df

    # filter user by type of categories
    def filter(self, df):
        usr_records, usr_index = {}, {}
        for j in df.index:
            if len(df.loc[j, 'category_code'].split('.')) >= self.filtering:
                usr = df.loc[j, 'user_id']
                usr_records.setdefault(usr, set())
                usr_records[usr].add(df.loc[j, 'category_code'])
                usr_index.setdefault(usr, [])
                usr_index[usr].append(j)

        idxes = []
        if usr_records != {}:
            for idx in [k for k, v in usr_records.items() if len(v) > 2]:
                idxes += usr_index[idx]

        return df.loc[idxes].copy(deep=True)


    # rebuild category_code
    def rebuild_category_code(self, df, df_len, PID):
        new_df = {}
        for column in df.columns:
            if column == 'category_code':
                category_level = 0
                category_len = 0
                
                for i in df.index:
                    if PID == 0:
                        print(f'\r[*] Progress: {round(round(i/df_len, 3)*100 , 2)}% [{i+1}/{df_len}]  ', end='')

                    # split category_code into multiple category_code_n (n >= 0)
                    contents = df.loc[i, column].split('.')

                    if len(contents) > category_level:
                        category_level = len(contents)
                        
                    for j in range(category_level):
                        new_df.setdefault(f'category_code_{j}', [])

                        # make sure every columns have same length
                        tmp = (category_len - len(new_df[f'category_code_{j}']))
                        if tmp > 0:
                            for n in range(tmp):
                                new_df[f'category_code_{j}'].append(pd.NA)
                        
                        if j < len(contents):
                            new_df[f'category_code_{j}'].append(contents[j])
                        else:
                            new_df[f'category_code_{j}'].append(pd.NA)

                    category_len += 1

            else:
                new_df.setdefault(column, [])
                new_df[column] = df[column].copy(deep=True)
        
        new_df = pd.DataFrame(new_df)

        return new_df

    def MP_rebuild_category_code(self, chunk_df, return_dict, PID):
        new_df = self.rebuild_category_code(chunk_df, chunk_df.shape[0], PID)
        return_dict[PID] = new_df
    
    # Generate a tree of category_code
    def tree(self, out, Dict, level):
        for k, v in Dict.items():
            out.write("\t"*level + f'L {k}\n')

            if type(v) == dict:
                self.tree(out, v, level + 1)
    
    # prepare for tree
    def tree_perpare(self, df):
        categories = {}
        category_codes = set()

        # category codes list
        cols = [col for col in df.columns if 'category_code_' in col]

        # category combination set
        cc_combination = set()

        total_len = df.shape[0]
        df.reset_index(drop=True, inplace=True)
        for i in df.index:
            print(f'\r[*] Progress: {round(round((i + 1)/total_len, 3)*100, 2)}% [{i+1}/{total_len}]  ', end='')
            contents = [df.loc[i, col] for col in cols]

            # record category codes in a set
            curr_combination = '-'.join(contents)
            for j in range(len(contents)):
                category_codes.add(contents[j])
            
            # optimization, skip category code combination that already processed.
            if curr_combination not in cc_combination:
                cc_combination.add(curr_combination)

                # create a curr_category and point to category dictionary - categories' location
                curr_category = categories
                for content in contents[:-1]:
                    curr_category.setdefault(content, {})

                    # if this category code is not the leaf
                    if curr_category[content] != '':
                        curr_category = curr_category[content]

                    else:
                        curr_category[content] = {}
                        curr_category = curr_category[content]

                if contents[-1] in curr_category:
                    # if category code is already exist and is a leaf, set its value to ''
                    if curr_category[contents[-1]] != {}:
                        curr_category.setdefault(contents[-1], '')
                else:
                    curr_category.setdefault(contents[-1], '')

        print(f'\n[*] Category codes({len(category_codes)}):')

        i = 1
        for category_code in category_codes:
            print(f'\t{i}.{category_code}')
            i += 1

        return categories
    
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
        for i in range(5):
            print(f'{df.iloc[i]}\n')
    
        print('------------------------------------------------------------------------')
    
    # MAIN
    def run(self):    
        start = time.time()

        print(f'[*] Reading from {os.path.basename(self.fi)}...')
        df = pd.read_csv(self.fi)
        self.preview(df)
        print('[*] Done.')

        df_len = df.shape[0]

        print(f'[*] Starting data preprocessing({df_len} records)...')
        chunk_iterator = [df.iloc[i:(i + self.chunksize)] for i in range(0, len(df), self.chunksize)]

        del df # release pressure

        i = 0
        processed_chunks = []
        for chunk_df in chunk_iterator:
            i += self.chunksize
            print(f'\r[*] Progress: {round(round((i+1)/df_len, 3)*100 , 2)}% [{i+1}/{df_len}]  ', end='')

            # drop columns
            chunk_df = self.drop_col(chunk_df)

            # drop N/A
            chunk_df.dropna(axis=0, inplace=True)

            # filter user
            chunk_df = self.filter(chunk_df)

            processed_chunks.append(chunk_df)

        print('\n[*] Done.')

        # save to local
        print('[*] concat data...')
        processed_df = pd.concat(processed_chunks, ignore_index=True)

        # save as csv file
        self.save2csv(processed_df, os.path.basename(self.fi).replace('.csv', '.clean.csv'))

        # save as HDF5 file
        # try:
        #     self.save2hdf5(processed_df, f"{os.path.basename(self.fi).replace('.csv', '.clean.split')}.h5")
        # except OverflowError as e:
        #     print(f'[*] Error: {e}')
        #     print('[*] Skip saving as HDF5 file.')

        self.preview(processed_df)

        # Optional
        if rebuild:
            print('[*] Starting rebuild category code...')
            processed_df.reset_index(drop=True, inplace=True)

            if self.multiProcessing > 1:
                self.chunksize = round(processed_df.shape[0]/self.multiProcessing) + 1
                chunk_iterator = [processed_df.iloc[i:(i + self.chunksize)] for i in range(0, len(processed_df), self.chunksize)]

                manager = mp.Manager()
                return_dict = manager.dict()

                df_len = processed_df.shape[0]
                processed_df = None

                i = 0
                processings = []
                speed_start = time.time()
                for chunk_df in chunk_iterator:
                    processing = mp.Process(target=self.MP_rebuild_category_code, args=(chunk_df, return_dict, i))
                    processing.start()
                    print(f'\n[*] Launch PROCESSING-{i}')
                    
                    processings.append(processing)
                    i += 1
                
                for j in range(len(processings)):
                    processings[j].join()
                    
                    if j == 0:
                        print(f'\n[*] PROCESSING-{j} done.')
                    else:
                        print(f'[*] PROCESSING-{j} done.')

                processed_chunks = []

                PIDs = sorted(return_dict)
                for PID in PIDs:
                    processed_chunks.append(return_dict[PID])
            
            else:
                speed_start = time.time()
                chunk_iterator = [processed_df.iloc[i:(i + self.chunksize)] for i in range(0, len(processed_df), self.chunksize)]
                
                df_len = processed_df.shape[0]
                processed_df = None

                i = 0
                processed_chunks = []
                for chunk_df in chunk_iterator:
                    # rebuild category code column
                    chunk_df = self.rebuild_category_code(chunk_df, df_len, 0)
                    processed_chunks.append(chunk_df)

                    i += self.chunksize
            
            del chunk_iterator # release pressure

            speed_time = time.time() - speed_start
            print(f'[*] Done, processing speed: {round(df_len/speed_time)} rows/s.')
            
            # cat
            print('[*] concat data...')
            processed_df = pd.concat(processed_chunks, ignore_index=True)

            # drop 1/3 category columns
            category_cols = [col for col in processed_df.columns if 'category_code_' in col]
            dropped_num = int(round(len(category_cols)/3, 0))
            category_dropped_cols = [f'category_code_{len(category_cols) - i - 1}' for i in range(len(category_cols))][0:dropped_num]
            
            print('[*] To avoid dataset in a small size after drop NA, we will drop 1/3 category columns.')
            print(f'[*] Dropping column: {category_dropped_cols}...')
            processed_df.drop(category_dropped_cols, axis=1, inplace=True)
            
            print('[*] Dropping NA...')
            processed_df.dropna(inplace=True)

            print('[*] Done.')

            # save to local
            self.save2csv(processed_df, os.path.basename(self.fi).replace('.csv', '.clean.split.csv'))

            # save as HDF5 file
            # try:
            #     self.save2hdf5(processed_df, f"{os.path.basename(self.fi).replace('.csv', '.clean.split')}.h5")
            # except OverflowError as e:
            #     print(f'[*] Error: {e}')
            #     print('[*] Skip saving as HDF5 file.')
            
            self.preview(processed_df)

        if self.rebuild:
            # Draw tree
            print('[*] Perpare data for plot category tree...')
            categories = self.tree_perpare(processed_df)
            fo = os.path.join(self.dir_o, f"{os.path.basename(self.fi).replace('.csv', '')}.tree.txt")

            print(f'[*] Generating tree file to {os.path.basename(fo)}...')
            with open(fo, 'w') as out:
                self.tree(out, categories, 1)
            print('[*] Done.')

        time_spend = time.time() - start
        if time_spend < 60:
            print(f'\n Time spend: {round(time_spend, 2)} second.\n')
        else:
            print(f'\n Time spend: {round((time_spend)/60, 2)} minutes.\n')


if __name__ == '__main__':
    help = \
'''
Example:
CONDITION 1
    rebuild_need_statistic.csv
    event_time,event_type,product_id,category_id,category_code,brand,price,user_id,user_session
    2019-10-01 00:00:00 UTC,view,3900821,2053013552326770905,appliances.environment.water_heater,aqua,33.2,554748717,9333dfbd-b87a-4708-9857-6336556b0fcc
    2019-10-01 00:00:10 UTC,view,28719074,2053013565480109009,apparel.shoes.keds,baden,102.71,520571932,ac1cd4e5-a3ce-4224-a2d7-ff660a105880
    ...

    python data_pipeline.py -I rebuild_need_statistic.csv -O path/to/directory -S 10000 -COL event_time,user_session -REBUILD T -MP 2
    [*] For such .csv whose category_code is not separated(appliances.environment.water_heater), -REBUILD need be T.
    [*] -S is optional, default is 10000. This means the chunk size processed each time, depend on your RAM.
    [*] -MP is optional, default 1. This means enable multi processings processing with 2 processings.

CONDITION 2
    statistic.csv
    event_time,event_type,product_id,category_id,category_code_0,category_code_1,category_code_2,brand,price,user_id,user_session
    2019-10-01 00:00:00 UTC,view,3900821,2053013552326770905,appliances,environment,water_heater,aqua,33.2,554748717,9333dfbd-b87a-4708-9857-6336556b0fcc
    2019-10-01 00:00:10 UTC,view,28719074,2053013565480109009,apparel,shoes,keds,baden,102.71,520571932,ac1cd4e5-a3ce-4224-a2d7-ff660a105880
    ...

    python data_pipeline.py -I statistic.csv -O path/to/directory -S 10000 -COL event_time,user_session -REBUILD F -MP 1
    [*] -REBUILD need be F as category_code is seperated. You can skip this as default is F.
    [*] -S is optional, default is 10000. This means the chunk size processed each time, depend on your RAM.

CONDITION 3
    path/to/my_statistics/
    python data_pipeline.py -I path/to/my_statistics/ -O path/to/directory -COL event_time,user_session -R T -MP 1
    [*] To process multiple files in a row, use -R T to achieve. This will preprocess all files within the path/to/my_statistics/ directory.
'''

    parser = argparse.ArgumentParser(description='DATA PREPROCESSING PIPELINE',
                                     epilog=help,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-I', '--INPUT', required=True, help='If you want to input a dictionary, plz type -R T as well')
    parser.add_argument('-O', '--OUTPUT_DIRECTORY', required=True, help='File name is not required, just need path.')
    parser.add_argument('-S', '--CHUNK_SIZE', default=10000, type=int, help='Set chunk size for processing data chunk by chunk, default is 10000.')
    parser.add_argument('-COL', '--DROPPED_COLUMNS', default=[], help='Optional, specify whcih column to drop, if there are multiple columns to drop, write as: col_A,col_B,...')
    parser.add_argument('-REBUILD', '--REBUILD_CATEGORY_CODE', default=False, choices=['T', 'F'], help='Optional, choose from T&F. This is just for data which category_code is not separated. Such as: electronics.smartphone')
    parser.add_argument('-R', '--RECURSIVE', default=False, choices=['T', 'F'], help='Optional, choose from T&F. This is for process multiple files in a row. If T, -I should be a directory path.')
    parser.add_argument('-MP', '--MULTI_PROCESSING', default=1, help='Optional, default is 1 processing. Entering the number of processing you want. If you specify processing number that larger than 1, chunk size will be assigned automaticly.')
    parser.add_argument('-F', '--FILTERING', default=2, type=int, help='Optional, default is 3. This is to check and remove the user whose type of categories is less than the input number')


    args = parser.parse_args()
    fi = args.INPUT
    dir_o = args.OUTPUT_DIRECTORY
    chunksize = args.CHUNK_SIZE
    dropped_cols = args.DROPPED_COLUMNS
    rebuild = args.REBUILD_CATEGORY_CODE
    recursive = args.RECURSIVE
    multiProcessing = int(args.MULTI_PROCESSING)
    filtering = args.FILTERING

    if recursive == 'T':
        recursive = True
    else:
        recursive = False
    
    # For processing files in a row
    if recursive:
        if not os.path.isdir(fi):
            sys.exit(f'[*] Error: you have use "-R T", but {dir_o} is not a directory!')

        if not os.path.exists(dir_o):
            os.mkdir(dir_o)

        if not os.path.isdir(dir_o):
            sys.exit(f'[*] Error: {dir_o} need to be a directory!')
        
        dropped_cols = [col for col in dropped_cols.split(',')]

        if rebuild == 'T':
            rebuild = True
        else:
            rebuild = False

        files = os.listdir(fi)
        i = 1
        for file in files:
            if file != '.DS_Store':
                curr_fi = os.path.join(fi, file)
                print(f'[*] Current file - {file}, {len(files) - i} remaining.')

                # MAIN
                preprocess = preprocessing(curr_fi, dir_o, chunksize, dropped_cols, rebuild, multiProcessing, filtering)
                preprocess.run()
            
            i += 1

    # For processing single file
    else:
        if not os.path.exists(fi):
            sys.exit(f'[*] Error: {fi} not exist!')
        
        if not os.path.isdir(dir_o):
            sys.exit(f'[*] Error: {dir_o} need to be a directory!')
        
        dropped_cols = [col for col in dropped_cols.split(',')]

        if rebuild == 'T':
            rebuild = True
        else:
            rebuild = False

        # MAIN
        preprocess = preprocessing(fi, dir_o, chunksize, dropped_cols, rebuild, multiProcessing, filtering)
        preprocess.run()
