import utils.io_manager as iom
import utils.data as dat
import variables as vrb
import utils.general as grl

import tensorflow as tf
tf.random.set_seed(vrb.RANDOM_SEED)

def get_timeseries_dataset_from_array(data_set, seq_length, prediction_interval, shuffle=False, seed=None):
    """
    Get Time Series dataset divided into batches

    Args:
    data_set (DataFrame): Original dataset
    seq_length (str): Sequence length
    prediction_interval (int): Number of predicted items
    shuffle (bool): True if shuffle data
    seed (int): Seed to replicate results
    """
    
    targets = data_set[vrb.COLUMN_CLOSE][seq_length:]
    if prediction_interval > 1:
        targets = None
        
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data_set.to_numpy(),
        targets=targets,
        sequence_length=seq_length,
        batch_size=32,
        shuffle=shuffle,
        seed=seed
    ) 

    if prediction_interval > 1:
        #TODO:: Future requirement mapping(ds)
        a = 0

    return ds

def get_timeseries_train_valid_test_sets(dataframe, columns, prediction_interval, seq_length=vrb.DEFAULT_SEQUENCE_LENGTH):
    """
    Get Time Series train, validation and test dataframes divided into batches

    Args:
    dataframe (DataFrame): Original dataset
    columns (array): Columns
    prediction_interval (int): Number of predicted items
    seq_length (str): Sequence length
    """
    
    tf.random.set_seed(vrb.RANDOM_SEED)

    df_train, df_valid, df_test = dat.get_train_valid_test_sets(dataframe, columns)

    train_ds = get_timeseries_dataset_from_array(df_train, seq_length, prediction_interval, shuffle=True, seed=vrb.RANDOM_SEED)
    valid_ds = get_timeseries_dataset_from_array(df_valid, seq_length, prediction_interval)
    test_ds = get_timeseries_dataset_from_array(df_test, seq_length, prediction_interval)

    return train_ds, valid_ds, test_ds
    

def add_model(series, name, model_type, prediction_interval, columns, units=1, seq_length=vrb.DEFAULT_SEQUENCE_LENGTH):
    """
    Create model and add to series

    Args:
    series (serie): Series to add
    name (str): Model name
    model_type(str): Model type (Dense, RNN, LSTM, GRU)
    prediction_interval (int): Number of predicted items
    columns (array): Columns
    units (int): Main layer number of units
    seq_length (str): Sequence length
    """
    tf.random.set_seed(vrb.RANDOM_SEED)

    layer = None
    match model_type:
        # case "dense":
        #     layer = tf.keras.layers.Dense(prediction_interval, input_shape=[seq_length])
        case "RNN":
            layer = tf.keras.layers.SimpleRNN(units)
        case "LSTM":
            layer = tf.keras.layers.LSTM(units)
        case "GRU":
            layer = tf.keras.layers.GRU(units)
    
    # Sequential model 
    model = tf.keras.Sequential()
    model.add( tf.keras.layers.Input(shape=(seq_length,len(columns))) )
    if layer != None:
        model.add(layer)
    model.add(tf.keras.layers.Dense(prediction_interval))

    features_type = grl.get_features_type(columns)

    key = f"{name}_{model_type}{units}_{features_type}{len(columns)}_out{prediction_interval}"
    print(f"Model created: {key} => {model.layers}")

    series[key] = model
    return model


def model_compile(model, learning_rate=0.02):
    """
    Model compilation

    Args:
    model (object): Model to compile
    learning_rate: Model compilation learning rate. Default=0.02
    """
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(), optimizer=opt, metrics=["mae","root_mean_squared_error"])

def model_fit(model, train_set, valid_set, epochs, patience, save_to=""):
    """
    Model fitting

    Args:
    model (object): Model to compile
    train_set (Dataframe): Train dataframe
    valid_set (Dataframe): Validation dataframe
    epochs (int): Number of fit epochs
    patience (int): Fitting patience. If None and epochs < 100, patience=epochs. If None and epochs >= 100, patience = 10% epochs
    save_to (str): If no blank, filename for model writting
    """
    patience = int(epochs//10) if patience == None else patience 
    if epochs < 100: 
        patience = epochs
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience= patience, restore_best_weights=True)

    print(f">>> Fitting model patience: {patience}")
    history = model.fit(train_set, validation_data=valid_set, epochs=epochs,
                        callbacks=[early_stopping_cb])

    if (save_to != ""):
        iom.save_model(model, save_to)
        
    return history

def model_evaluate(model, valid_set):
    """
    Model evaluation with validation set

    Args:
    model (object): Model to compile
    valid_set (Dataframe): Validation dataframe
    """
    valid_loss, valid_mae, valid_rmse = model.evaluate(valid_set)
    return valid_loss, valid_mae, valid_rmse

def fit_and_evaluate(model, name, columns, train_set, valid_set, learning_rate=0.02, epochs=500, patience = None, force_compilation=False):
    """
    Model fitting and evaluation with validation set

    Args:
    model (object): Model to compile
    name (str): Model name
    train_set (Dataframe): Train dataframe
    valid_set (Dataframe): Validation dataframe
    learning_rate: Model compilation learning rate. Default=0.02
    epochs (int): Number of fit epochs. Default 500
    patience (int): Fitting patience. If None and epochs < 100, patience=epochs. If None and epochs >= 100, patience = 10% epochs
    force_compilation (bool): If True and model file exists, force model compilation
    """

    time_start = grl.get_datetime()

    model_state = ""
    current_model = iom.read_model(name)
    if current_model == None or force_compilation:
        current_model = model
        model_compile(current_model, learning_rate)
        model_state = "new compilation"
        print(f">>> Model compiled: '{name}'")
    else:
        model_state = "loaded"
        print(f">>> Model exists: '{name}'")

    history = model_fit(current_model, train_set, valid_set, epochs=epochs, patience=patience, save_to=name)

    valid_loss, valid_mae, valid_rmse = model_evaluate(current_model, valid_set)
    print(f">>> Model MAE '{name}': {valid_mae}")

    iom.add_fit_info(name, model_state, columns, time_start, epochs, len(history.epoch), valid_loss, valid_mae, valid_rmse)
    
    return history, valid_loss, valid_mae


def train_models(df, columns_array, prediction_interval, epochs):
    """
    Executing models with different columns sets

    Args:
    df (DataFrame): Source dataframe
    columns_array ([str[]]): Columns sets
    prediction_interval: Number of predicted items
    epochs: Fit epochs
    """
    for columns in columns_array:
        train_ds, valid_ds, test_ds = get_timeseries_train_valid_test_sets(df, columns, prediction_interval)

        serie_models = {}
        _ = add_model(serie_models, "model_1", "Dense", prediction_interval, columns)
        _ = add_model(serie_models, "model_2", "RNN", prediction_interval, columns, units=1)
        _ = add_model(serie_models, "model_3", "RNN", prediction_interval, columns, units=32)
        _ = add_model(serie_models, "model_4", "LSTM", prediction_interval, columns, units=32)
        _ = add_model(serie_models, "model_5", "GRU", prediction_interval, columns, units=32)

        for id, key in enumerate(serie_models):
            print(f"{id}: MODEL [{key}]")
            print(f" - Columns: {columns}")
            history, valid_loss, valid_mae = fit_and_evaluate(serie_models[key], key, columns, train_ds, valid_ds, epochs=epochs)
            print()


def get_series_columns(df_original, prediction_interval):
    """
    Get timeseries width different set of columns

    Args:
    df_original (DataFrame): Source dataframe
    prediction_interval: Number of predicted items
    """
    set_series = {}
    set_columns = {}

    set_columns[1] = [dat.TARGET]
    set_series[1] = get_timeseries_train_valid_test_sets(df_original, set_columns[1], prediction_interval)

    set_columns[2] = dat.get_colums_original_features_2()
    set_series[2] = get_timeseries_train_valid_test_sets(df_original, set_columns[2], prediction_interval)

    set_columns[5] = dat.get_columns_sizes_features_5()
    set_series[5] = get_timeseries_train_valid_test_sets(df_original, set_columns[5], prediction_interval)

    set_columns[11] = dat.get_columns_all_features_11(df_original)
    set_series[11] = get_timeseries_train_valid_test_sets(df_original, set_columns[11], prediction_interval)

    return set_series, set_columns

def get_model_info(name):
    """
    Return model configuration based on model name (filename)

    Args:
    name (str): Model name
    """

    str_array = name.split("_")
    model_name = str_array[2]

    features_str = str_array[3][str_array[3].index("var"): ]
    features_num = int(features_str.replace("var",""))

    # out_str = name[name.index("_out"):]
    out_num = int(str_array[4].replace("out",""))

    return model_name, features_num, out_num