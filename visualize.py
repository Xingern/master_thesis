import pandas as pd
import plotly.figure_factory as ff

def print_df_info(df):
    """
    Prints the number of rows of input DataFrame, number of rows without NaN
    and negative values.
    """
    nan_count = (df.count(axis=1) == len(df.columns)).sum()
    neg_count = ((df.iloc[:, 1:] > 0).all(1)).sum()
    
    print("Size of the DataFrame:", df.__len__())
    print("Number of rows without NaN values:", nan_count)
    print("Number of rows without negative values:", neg_count)
    
def generate_overview(df):
    """
    Returns a df consisting of NaN counter and output from 
    DataFrame.describe()
    """
    nan_count_per_column = df.isna().sum()
    describe_df = df.describe()
    overview_df = pd.concat([pd.DataFrame(nan_count_per_column).T, 
                             describe_df], 
                            axis=0)
    overview_df = overview_df.rename(index={0: 'NaN count', 
                                            'count': 'Valid count'})
    return overview_df.round(2)

def make_heatmap(df):
    """
    Takes in a correlation matrix, df, and returns a heatmap
    """
    cell_height=50
    fig_height = len(df) * cell_height

    heatmap = ff.create_annotated_heatmap(
        z=df.values,
        x=list(df.columns),
        y=list(df.index),
        annotation_text=df.round(2).values,
        colorscale='Viridis'
    )

    heatmap.update_layout(height=fig_height)
    return heatmap