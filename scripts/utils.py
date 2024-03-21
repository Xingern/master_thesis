import os
import matplotlib.pyplot as plt
import pandas as pd

def save_file(obj, file_path, filename, file_type, overwrite=False):
    """
    Save a plot or DataFrame to a file. Supported formats are PDF for plots 
    and pickle for DataFrames.

    Parameters
    ----------
        obj (plotly.graph_objs._figure.Figure, matplotlib.figure.Figure or pandas.DataFrame): 
            The figure to save.
            
        file_path (str): 
            The path to the directory where the plot should be saved.
            
        filename (str): 
            The name of the file to save the plot as.
            
        file_type (str): 
            The type of plot ('plotly' or 'matplotlib').
            
        overwrite (bool): 
            Whether to overwrite an existing file. Defaults to False.
            
    Examples
    --------
    For Matplotlib:
    >>> save_file(plt.gcf(), 'path/to/plots', 'my_plot.pdf', 'matplotlib')

    For Plotly:
    >>> save_file(fig, 'path/to/plots', 'my_plot.pdf', 'plotly')
    
    For Pandas DataFrame:
    >>> save_file(df, 'path/to/data', 'my_data.pickle', 'pandas')
    """
    os.makedirs(file_path, exist_ok=True)
    file_path = os.path.join(file_path, filename)
    if not os.path.exists(file_path) or overwrite:
        if file_type == "plotly":
            obj.write_image(file_path)
        elif file_type == "matplotlib":
            obj.savefig(file_path, format="pdf", bbox_inches='tight')
        elif file_type == "pandas":
            obj.to_pickle(file_path)
        else:
            print("Plot type not recognized. Use 'plotly', 'matplotlib', or 'pandas'.")
            return
        
        print(f"File {filename} was saved to {file_path}.")
    else:
        print(f"File {filename} already exists at {file_path}. Set 'overwrite=True' to overwrite the file.")
