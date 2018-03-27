import tensorflow as tf
import shutil
import os



_CSV_COLUMNS = [
    'VISA_CLASS', 'EMPLOYER_NAME', 'EMPLOYER_STATE',
    'EMPLOYER_COUNTRY', 'SOC_NAME', 'NAICS_CODE', 'TOTAL_WORKERS', 'FULL_TIME_POSITION',
    'PREVAILING_WAGE', 'PW_UNIT_OF_PAY','PW_SOURCE','WAGE_RATE_OF_PAY_FROM',
    'WAGE_RATE_OF_PAY_TO',
    'WAGE_UNIT_OF_PAY','H_1B_DEPENDENT','WORKSITE_STATE','CASE_STATUS']

_CSV_COLUMN_DEFAULTS = [['N/A'],['N/A'],['N/A'],
                        ['N/A'],['N/A'],['N/A'],[0],['N/A'],
                        [0],['N/A'], ['N/A'], [0],
                        [0],
                        ['N/A'], ['N/A'], ['N/A'], [0]]

def parse_csv(data_file):
    print("Parsing" + data_file)
    columns = tf.decode_csv(data_file, record_defaults=_CSV_COLUMN_DEFAULTS)
    for column in columns:
        print("column name : "+str(column)+" type : "+str(column.dtype))
    assert columns[6].dtype == tf.int32

if __name__ == '__main__':
    data_file = "F:\\Coding\\Python\\AIAM\\data\\train.csv"
    parse_csv(data_file)

