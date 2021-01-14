## Application of Naïve Bayes and Random Forest on Classification of Potential Bike Customer 

This project aims to predict and classify potential bike buyers from the existing customer database of a company based on the given demographic features such as age, accommodation, income, gender etc. Therefore, this project compares the performance of two classification algorithms (**Naïve Bayes & Random Forest**)** on binary class classification. **A poster** is created for interpretation and critical analysis.

In detailed implementation, data-preprocessing and wrangling are performed in Python and model fittings are performed in MATLAB. The requred software versions are as follows:
- MATLAB ver. R2020a
- MATLAB TOOLBOX - Statistics and Machine Learning Toolbox
- Python (> Ver. 3)

Model training process and intermediate results please refer to '*Intermediate Results*' and '*Glossary*'.

### Data Source: 
Bike Buying Prediction For Adventure Works Cycles (Microsoft EDX competition data-set), available at: https://www.kaggle.com/rahulsah06/bike-buying-prediction-for-adventure-works-cycles?select=AW_BikeBuyer.csv

Three datasets are downloaded and available in the following folder: 
- 'Data\AdvWorksCusts.csv'
- 'Data\AW_AveMonthSpend.csv'
- 'Data\AW_BikeBuyer.csv'

Datasets are wrangled and merged by python file (@'Data\Clean_Bike_Datasets.py') and cleaned dataset for modelling is avaiable in 'Data\BikeMergeClean.csv'

####  Description of MATLAB Files
1) Final trained models, including the tuned parameters/ hyperparameter ('*.mat' File)
    a) 'Model_NB_parameter.mat'
    b) 'Model_RF_parameter.mat'

2) Script to Conduct Initial Analysis on the Dataset ('*m.' file)
    a) 'Initial_Analysis_On_Data_Distribution.m'

3) Script to evaluate Validation Method parameters and conduct Data Partitioning ('*m.' file)
    a) 'EvaluationMethodParameter.m'

4) Self-Defined Function to Calculate F1 Score for convenience ('*m.' file)
    a) 'f1score.m'

5) Scripts are run to test the final trained models and process ('*m.' file)
- Import the trained model ('*mat' files in [1]) 
- Import Final Test Data for testing ('*m' file in [3])
- Test loaded trained models on final test data 
- Calculate performance metrics e.g. Error rate, Area Under Curve (AUC) under Precision-Recall Curve, F1 Score
- Plot Graph against desired features
- Two scripts as follows:
    a) 'RF_Final_Model.m' 
    b) 'NB_Final_Model.m'

6) Model training, tuning and hyperparameter optimisation experiments scripted separately ('*m.' file). Best results are saved in [5]
   a) 'NB_Model_Selection.m'
   b) 'RF_Model_Selection.m'


7) For detailed assessment results during the training period in [6], results are saved in:
    a) Naive Bayes assessment results: 'NB_Parameter_Assessment\NB_Model_all_varaible.mat'
    b) Random Forest assessment results: 'RF_Parameter_Assessment\*mat' --all mat files

8) For detailed assessment plots during the training period in [6] & [7], charts are saved in 'Chart' folder


