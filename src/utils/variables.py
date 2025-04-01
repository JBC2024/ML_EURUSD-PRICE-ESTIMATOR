RANDOM_SEED = 808

DATASOURCE_PATH = "../data/EURUSD-H1-201701_202502.csv"
FIT_RESULTS_PATH = "../data/history/fit_info.csv"
TEST_RESULTS_PATH = "../data/history/test_info.csv"
SAVED_MODELS_DIRECTORY = "../models/"

EURUSD_DECIMALS = 6

COLUMN_DATE_ORIGINAL = "DateTime"
COLUMN_DATE = "Date"
COLUMN_OPEN = "Open"
COLUMN_HIGH = "High"
COLUMN_LOW = "Low"
COLUMN_CLOSE = "Close"
COLUMN_VOLUME = "Volume"

COLUMN_BODY = "Body"
COLUMN_UPPER_SHADOW = "UpperShadow"
COLUMN_LOWER_SHADOW = "LowerShadow"
COLUMN_DAY_OF_WEEK = "DayOfWeek"

INFO_COLUMN_NAME = "Model_Name"
INFO_COLUMN_FEATURES_TPYE = "Features_Type"
INFO_COLUMN_STATE = "Model_State"
INFO_COLUMN_COLUMNS = "Dataset_Columns"
INFO_COLUMN_EPOCHS_CONFIG = "Epochs Total"
INFO_COLUMN_EPOCHS_COMPLETED = "Epochs Completed"
INFO_COLUMN_START_DATE = "Date_Start"
INFO_COLUMN_FINISH_DATE = "Date_Finish"
INFO_COLUMN_TOTAL_TIME = "Total_Time"
INFO_COLUMN_VALID_LOSS = "Valid_Loss"
INFO_COLUMN_VALID_MAE = "Valid_MAE"
INFO_COLUMN_VALID_RMSE = "Valid_RMSE"

COLUMNS_INFO_ARRAY = [INFO_COLUMN_NAME, 
                      INFO_COLUMN_FEATURES_TPYE,
                      INFO_COLUMN_STATE,
                      INFO_COLUMN_COLUMNS,
                      INFO_COLUMN_START_DATE,
                      INFO_COLUMN_FINISH_DATE,
                      INFO_COLUMN_TOTAL_TIME,
                      INFO_COLUMN_EPOCHS_CONFIG,                   
                      INFO_COLUMN_EPOCHS_COMPLETED,
                      INFO_COLUMN_VALID_LOSS,
                      INFO_COLUMN_VALID_MAE,
                      INFO_COLUMN_VALID_RMSE
                      ]

DEFAULT_SEQUENCE_LENGTH = 48