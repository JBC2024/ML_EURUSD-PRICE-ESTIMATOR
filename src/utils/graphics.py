import plotly.graph_objects as go
import variables as vrb

def paint_candlestick_graph(datasource):
    """
    Paint candlestick graph

    Args:
    datasource (dataframe): Data
    """
    fig = go.Figure(data=[go.Candlestick(
                        x=datasource[vrb.COLUMN_DATE],
                        open=datasource[vrb.COLUMN_OPEN],
                        high=datasource[vrb.COLUMN_HIGH],
                        low=datasource[vrb.COLUMN_LOW],
                        close=datasource[vrb.COLUMN_CLOSE])
                    ])
    fig.show();

