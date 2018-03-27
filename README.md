# Tensorflow_samples
The respository contains deep learning classifier from tensorflow 1.5 for predicting the H1B visa status - "Certified", "Withdrawn","Denied", "CertifiedWithdrawn"

# Data
The data used for building the models is from kaggle - https://www.kaggle.com/trivedicharmi/h1b-disclosure-dataset . There has been a R code to modify the data to be used in the model accordingly. The R code will be uploaded shortly.

# Codes
The tensorflow codes have been implemented in python. visa_classifier.py in the codes directory is the primary python file for running the deep learning networks. The models are evaluated on both training and test data to evaluate bias and variance.

# Models
3 types of models can be built using the python codes. They are the wide model, the deep model and the hybrid wide and deep model. More information of the models can be found at https://www.tensorflow.org/tutorials/wide_and_deep

# Configuration
The python code can be controlled from the config.txt file in the config folder. The config file has the following fields
1. HIDDEN_UNITS : This property sets the hidden units. The number of hidden units per layer are comma separated. For example, 50,50,50,50
2. ROOT_DIR : This property sets the root or main directory  of the project. All the other directories like data, config etc. are derived from the root directory
3. TRAIN_DATA_FILE : Location of the training data file
4. TEST_DATA_FILE : Location of the test data file
5. RESULT_FILE : Location of the result file
6. MODEL_DIR : Location of the directory where the learnt model can be saved. This saved model can be used for visualization through Tensorboard. For more reference, please use https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard
7. MODEL_TYPE : This specifies the type of the model - wide, deep or the hybrid wide-deep
8. NUM_EPOCHS : Number of epochs that is number of times the entire training data set will be traversed
9. EPOCHS_PER_EVAL : Number of epochs to be used per evaluation.
10. BATCH_SIZE : Batch size of the training or test data set to be used for the model training. This is used for parallelization.

# Results
The results are saved in the result.txt file in the result directory. 
Sample result is as mentioned below. For each evaluation, the trained model is evaluated over both training and test data to assess the nature of the bias and the variance.

=================Iteration starts =============== 
Hidden Layer Structure : [50, 50, 50, 50]
****************************************************************************************************
Evaluation on the training set for bias 
Results at epoch (Training data)1
------------------------------------------------------------
accuracy : 0.51782566
average_loss : 348.2177
global_step : 5
loss : 6868385.5
****************************************************************************************************
Evaluation on the test set for variance 
Results at epoch (Test data)1
------------------------------------------------------------
accuracy : 0.3242723
average_loss : 524.5638
global_step : 5
loss : 7358756.0
===================Iteration ends =============== 

