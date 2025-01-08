from collections import Counter
from tqdm import tqdm
import pandas as pd
import numpy as np
import statistics
import argparse
import os

descriptions = \
"""
STEP 3
PARAMETER CALCULATION

Parameter calculation is to calculate the parameter of each customers' data
Include: 
- average_time_float: The average amount of time a user has been active, expressed as a float
- purchase_ration: the percentage of behavior 'purchase' in one user's records
- brand_loyalty_ratio: the ratio of same brand in one user's records
- most_freq_category: The three most frequently occurring commodity groups
- average_price: Average price of items viewed, added or purchased by users
"""

class data_statistic_transformation:
    def __init__(self, fi, sep, round_decimal, transformed_usr_records):
        self.fi = fi
        self.fo = fo
        self.sep = sep
        self.round_decimal = round_decimal
        self.transformed_usr_records = transformed_usr_records

    def cal_purchase_ratio(self, col):
        return round(col.tolist().count("purchase")/len(col), self.round_decimal)

    def cal_brand_loyalty_ratio(self, col):
        return round(col.tolist().count(statistics.mode(col))/len(col), self.round_decimal)

    def activity_category_distribution(self, col):
        counter = Counter(col)
        activity_distribution = {k: round(v/len(col), 1) for k, v in counter.items()}
        return activity_distribution

    def transform(self):
        df = pd.read_csv(self.fi, sep=self.sep)

        self.transformed_usr_records["average_time_float"].append(round(np.mean(df["time_float"]), self.round_decimal))
        self.transformed_usr_records["purchase_ratio"].append(self.cal_purchase_ratio(df["event_type"]))
        self.transformed_usr_records["average_price"].append(round(np.mean(df[["price"]]), self.round_decimal))
        self.transformed_usr_records["brand_loyalty_ratio"].append(self.cal_brand_loyalty_ratio(df["brand"]))

        # Find the top 3 most frequent values for category_code_2
        top_categories = df["category_code"].value_counts().head(3).index.tolist()
        while len(top_categories) < 3:  # Ensure the list always has 3 elements
            top_categories.append(None)  # Fill with None if fewer than 3 categories

        self.transformed_usr_records["most_freq_category_1"].append(top_categories[0])
        self.transformed_usr_records["most_freq_category_2"].append(top_categories[1])
        self.transformed_usr_records["most_freq_category_3"].append(top_categories[2])

        activity_distribution = self.activity_category_distribution(df["category_code"])

        self.transformed_usr_records["category_1_activity_weight"].append(activity_distribution[top_categories[0]])
        self.transformed_usr_records["category_2_activity_weight"].append(activity_distribution[top_categories[1]])
        self.transformed_usr_records["category_3_activity_weight"].append(activity_distribution[top_categories[2]])

        # self.transformed_usr_records["records_number"].append(df.shape[0])

        return self.transformed_usr_records

def save2csv(transformed_usr_records, fo):
    final_df = pd.DataFrame(transformed_usr_records)
    # final_df = final_df.sort_values(by=['records_number'], ascending=False)
    final_df.to_csv(fo, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DATA STATISTIC TRANSFORMATION',
                                     epilog=descriptions,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-I", "--INPUT_DIRECTORY", help="Path of user records directory. Each user records should be a independent file.", required=True)
    parser.add_argument("-O", "--OUTPUT_FILE", help="Path of output file with file name.", required=True)
    parser.add_argument("-S", "--SEPARATION", help="Separator of csv file, both input and output. If 'tab', type tab.", default=",")
    parser.add_argument("-R", "--ROUND_DECIMAL", help="The number of round decimal.", default=2)

    args = parser.parse_args()
    di = args.INPUT_DIRECTORY
    fo = args.OUTPUT_FILE
    sep = args.SEPARATION
    round_decimal = args.ROUND_DECIMAL

    if sep.lower() == 'tab':
        sep = '\t'

    if not fo.endswith('.csv'):
        fo += '.csv'

    # Initrialize
    transformed_usr_records = {
        "user_id": [],
        "average_time_float": [],
        "purchase_ratio": [],
        "average_price": [],
        "brand_loyalty_ratio": [],
        "most_freq_category_1": [],
        "category_1_activity_weight": [],
        "most_freq_category_2": [],
        "category_2_activity_weight": [],
        "most_freq_category_3": [],
        "category_3_activity_weight": [],
        # "records_number": [],
    }
    files = os.listdir(di)

    print(f'[*] Transforming user records within {di}:')
    for i in tqdm(range(len(files))):
        curr_fi = os.path.join(di, files[i])
        usr_id = files[i].split('.')[0]

        transformed_usr_records["user_id"].append(usr_id)

        data_transformer = data_statistic_transformation(curr_fi,
                                                         sep,
                                                         round_decimal,
                                                         transformed_usr_records)
        transformed_usr_records = data_transformer.transform()

    print('[*] Done.')

    # Save to local
    print(f'[*] Saving transformed user records to {fo}...')
    save2csv(transformed_usr_records, fo)
    print('[*] Done.')


