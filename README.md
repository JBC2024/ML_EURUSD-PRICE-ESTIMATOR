# EURUSD Price Estimator

📌 Author:  Justo Barco 

🎯 Objetivo del Proyecto:

The objective of the project is to estimate the closing price of the EUR-USD currency exchange based on the closing price of the previous hours.

🛠 Dataset Info:

- Downloadable under subscription (private)
- Columns
    - Open: First price within the timeframe
    - High: Maximum price reached within the timeframe
    - Low: Minimun price reached within the timeframe
    - Close: Last price within the timeframe
    - Volume: Number of assets that have been bought or sold within the timeframe

🛠 Solution:

    - Solution based on Deep Learning models (RNN, LSTM, GRU...)

🛠 Directory info:

    - src 
        - data: Data directory
            - history: Evaluation results
        - models: Saved models
        - notebooks: ipynb files
        - utils: functions and utils (py files)