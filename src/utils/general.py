import datetime as dt

def get_datetime():
    """
    Returns current datetime

    """
    return dt.datetime.now()


def get_features_type(columns):
    """
    Return features type (univar|multivar) based on the number of columns

    Args:
    filename (str): Filename to check
    """
    result = "univar"
    if len(columns) > 1:
        result = "multivar"
    return result
