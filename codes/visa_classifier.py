
import tensorflow as tf
import shutil
import os



_CSV_COLUMNS = [
    'VISA_CLASS', 'EMPLOYER_NAME', 'EMPLOYER_STATE',
    'EMPLOYER_COUNTRY', 'SOC_NAME', 'NAICS_CODE','FULL_TIME_POSITION',
    'PREVAILING_WAGE', 'PW_UNIT_OF_PAY','PW_SOURCE','WAGE_RATE_OF_PAY_FROM',
    'WAGE_RATE_OF_PAY_TO','TOTAL_WORKERS','WILLFUL_VIOLATOR',
    'WAGE_UNIT_OF_PAY','H_1B_DEPENDENT','WORKSITE_STATE','CASE_STATUS']

_CSV_COLUMN_DEFAULTS = [['N/A'],['N/A'],['N/A'],
                        ['N/A'],['N/A'],['N/A'],['N/A'],
                        [0],['N/A'], ['N/A'], [0],
                        [0],[0],['N/A'],
                        ['N/A'], ['N/A'], ['N/A'], ['N/A']]

LABEL_VOCABULARY = ["WITHDRAWN","CERTIFIEDWITHDRAWN","DENIED","CERTIFIED"]
HIDDEN_UNITS = [50, 50, 50, 50]
TRAIN_DATA_FILE = ""
TEST_DATA_FILE = ""
RESULT_FILE = ""





def read_input_data(data_file,num_epochs,shuffle,batch_size):
    assert tf.gfile.Exists(data_file), ('%s not found.' % data_file)

    def parse_csv(value):
        print("Parsing " + data_file)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('CASE_STATUS')
        return features, labels

    dataset = tf.data.TextLineDataset(data_file).skip(1)
    dataset = dataset.map(parse_csv, num_parallel_calls=4)
    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    print("Read dataset properly \n")
    return dataset



def transform_and_select_features():
    #Create categorical features
    visa_class = tf.feature_column.categorical_column_with_hash_bucket(
        'VISA_CLASS', hash_bucket_size=100)
    employer_name = tf.feature_column.categorical_column_with_hash_bucket(
        'EMPLOYER_NAME', hash_bucket_size=10000)
    employer_state = tf.feature_column.categorical_column_with_hash_bucket(
        'EMPLOYER_STATE', hash_bucket_size=1000)
    employer_country = tf.feature_column.categorical_column_with_hash_bucket(
        'EMPLOYER_COUNTRY', hash_bucket_size=1000)
    soc_name = tf.feature_column.categorical_column_with_hash_bucket(
        'SOC_NAME', hash_bucket_size=1000)
    naics_code = tf.feature_column.categorical_column_with_hash_bucket(
        'NAICS_CODE', hash_bucket_size=1000)
    full_time_position = tf.feature_column.categorical_column_with_hash_bucket(
        'FULL_TIME_POSITION', hash_bucket_size=1000)
    pw_unit_of_pay = tf.feature_column.categorical_column_with_hash_bucket(
        'PW_UNIT_OF_PAY', hash_bucket_size=1000)
    pw_source = tf.feature_column.categorical_column_with_hash_bucket(
        'PW_SOURCE', hash_bucket_size=1000)
    wage_unit_of_pay = tf.feature_column.categorical_column_with_hash_bucket(
        'WAGE_UNIT_OF_PAY', hash_bucket_size=1000)
    dependent = tf.feature_column.categorical_column_with_hash_bucket(
        'H_1B_DEPENDENT', hash_bucket_size=1000)
    worksite_state = tf.feature_column.categorical_column_with_hash_bucket(
        'WORKSITE_STATE', hash_bucket_size=1000)
    willful_violator = tf.feature_column.categorical_column_with_hash_bucket(
        'WILLFUL_VIOLATOR', hash_bucket_size=1000)

    #Create numeric features
    total_workers = tf.feature_column.numeric_column('TOTAL_WORKERS')
    prevailing_wage = tf.feature_column.numeric_column('PREVAILING_WAGE')
    wage_rate_of_pay_from = tf.feature_column.numeric_column('WAGE_RATE_OF_PAY_FROM')
    wage_rate_of_pay_to = tf.feature_column.numeric_column('WAGE_RATE_OF_PAY_TO')

    base_columns = [total_workers,prevailing_wage,wage_rate_of_pay_from,wage_rate_of_pay_to,visa_class,employer_name,employer_state,
                    employer_country,soc_name,naics_code,full_time_position,pw_unit_of_pay,pw_source,wage_unit_of_pay,dependent,
                    worksite_state,willful_violator]

    #crossed_columns = [tf.feature_column.crossed_column(['education', 'occupation'], hash_bucket_size=1000),
    #    tf.feature_column.crossed_column(
    #        [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
    #]

    deep_columns = [total_workers,prevailing_wage,wage_rate_of_pay_from,wage_rate_of_pay_to,
                    tf.feature_column.indicator_column(visa_class),tf.feature_column.indicator_column(employer_name),
                    tf.feature_column.indicator_column(employer_state),
                    tf.feature_column.indicator_column(employer_country),tf.feature_column.indicator_column(soc_name),
                    tf.feature_column.indicator_column(naics_code),tf.feature_column.indicator_column(full_time_position),
                    tf.feature_column.indicator_column(pw_unit_of_pay),tf.feature_column.indicator_column(pw_source),
                    tf.feature_column.indicator_column(wage_unit_of_pay),tf.feature_column.indicator_column(dependent),
                    tf.feature_column.indicator_column(worksite_state),
                    tf.feature_column.indicator_column(willful_violator)]

    return base_columns,deep_columns



def build_estimator(model_dir, model_type):
    """Build an estimator appropriate for the given model type."""
    wide_columns, deep_columns = transform_and_select_features()
    # Defined 4 hidden unit layers with 50,30,20 and 10 units
    #hidden_units = [50, 30, 20, 10]

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))

    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns,
            config=run_config,label_vocabulary=LABEL_VOCABULARY,n_classes=4)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=HIDDEN_UNITS,
            config=run_config,label_vocabulary=LABEL_VOCABULARY,n_classes=4)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=HIDDEN_UNITS,
            config=run_config,label_vocabulary=LABEL_VOCABULARY,n_classes=4)


def run_classifier(unused):
    # Clean up the model directory if present
    train_data_file = "F:\\Coding\\Python\\AIAM\\data\\train.csv"
    test_data_file = "F:\\Coding\\Python\\AIAM\\data\\test.csv"
    result_data_file = "F:\\Coding\\Python\\AIAM\\results\\results.txt"
    num_epochs = 3
    epochs_per_eval = 1
    batch_size = 20000
    model_dir="F:\\Coding\\Python\\AIAM\\model\\"
    model_type="deep_wide_combo"
    shutil.rmtree(model_dir+model_type, ignore_errors=True)
    model = build_estimator(model_dir+model_type, model_type)

    writer = open(result_data_file,'a')
    writer.write("=================Iteration starts =============== \n")
    # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
    writer.write("Hidden Layer Structure : "+str(HIDDEN_UNITS)+"\n")
    for n in range(int(num_epochs/epochs_per_eval)):
        #print("Searching training file at "+os.getcwd()+"\\"+train_data_file)
        model.train(input_fn=lambda: read_input_data(train_data_file, epochs_per_eval, True, batch_size))
        #print("Searching test file at "+os.getcwd()+"\\"+test_data_file)
        writer.write("*"*100)
        writer.write("\n")
        writer.write("Evaluation on the training set for bias")
        results = model.evaluate(input_fn=lambda: read_input_data(train_data_file, 1, False, batch_size))
        # Display evaluation metrics on training data
        writer.write('Results at epoch (Training data)'+str((n + 1) * epochs_per_eval)+"\n")
        writer.write('-' * 60)
        writer.write("\n")
        for key in sorted(results):
            writer.write(str(key) + " : " + str(results[key]) + "\n")
        writer.write("*"*100)
        writer.write("\n")
        writer.write("Evaluation on the test set for variance \n")
        results = model.evaluate(input_fn=lambda: read_input_data(test_data_file, 1, False, batch_size))
        # Display evaluation metrics on test data
        writer.write('Results at epoch (Test data)'+str((n + 1) * epochs_per_eval)+"\n")
        writer.write('-' * 60)
        writer.write("\n")
        for key in sorted(results):
            writer.write(str(key)+" : "+str(results[key])+"\n")
    writer.write("===================Iteration ends =============== \n")
    writer.close()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=run_classifier)

