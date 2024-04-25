import os
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# Global settings for plots
plt.style.use('fivethirtyeight') 
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['figure.facecolor'] = 'white'  # For the figure background
plt.rcParams['axes.facecolor'] = 'white'    # For the axes background
plt.rcParams['axes.edgecolor'] = 'white'    # Set the axes edge color to white
plt.rcParams['savefig.facecolor'] = 'white' # For the figure background when saving
pd.set_option('display.max_columns', None)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] 

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
        elif file_type == "matplotlib-png":
            obj.savefig(file_path, format="png", bbox_inches='tight')
        elif file_type == "pandas":
            obj.to_pickle(file_path)
        else:
            print("Plot type not recognized. Use 'plotly', 'matplotlib', or 'pandas'.")
            return
        
        print(f"File {filename} was saved to {file_path}.")
    else:
        print(f"File {filename} already exists at {file_path}. Set 'overwrite=True' to overwrite the file.")

def make_evaluation_plots(df_train, df_test, name, plots_path, overwrite, limit, error_line=0.01, res_limit=[-0.3, 0.3], mean=3.85, gridsize=1000):
    """
    Main function to create all evaluation plots for a model whcih includes
    parity plot, residual plot, and distribution plot.
    
    Parameters
    ----------
    df_train : pd.DataFrame
        The DataFrame with the training set data. Use generate_prediction_df() to create.
        
    df_test : pd.DataFrame
        The DataFrame with the testing set data. Use generate_predictio_df() to create.
        
    name : str
        The name of the model for the plots. Will be used in the filename.
    
    plots_path : str
        The path to the directory where the plots should be saved.
        
    overwrite : bool
        Whether to overwrite existing files.
        
    limit : list
        The limits for the x and y axes that are plotted.
        
    error_line : float
        The error line to plot on the parity plot. Defaults to 0.01. Can either
        be 0.01, 0.03, or 0.05.
        
    res_limit : list
        The limits for the x and y axes that are plotted on the residual plot.
        
    mean : float
        The mean value of the residuals. Defaults to 3.85.
        
    gridsize : int
        The number of grid points to use for the distribution plot. Defaults to 1000.
        
    Examples
    --------
    plots_path = "/Users/junxingli/Desktop/master_thesis/figs/ann/stationary-whole-V3/" 
    >>> make_evaluation_plots(plots_path, 
    >>>                   overwrite=False, 
    >>>                   limit=[3.5, 4.4], 
    >>>                   error_line=0.05, 
    >>>                   res_limit=[-0.3, 0.3], 
    >>>                   mean=df['SRD'].mean())
        
        
       
    """
    plot_parity(df_train, df_test, name, plots_path, overwrite, limit, error_line)
    plot_residuals(df_train, df_test, name, plots_path, overwrite, limit, error_line, res_limit, mean)
    #residual_histogram_plot(df_train, df_test, name, plots_path, overwrite, error_line, res_limit, mean)
    plot_distribution(df_train, df_test, name, plots_path, overwrite, limit, gridsize)
    
def plot_parity(df_train, df_test, name, plots_path, overwrite, limit, error_line):
    """
    Generate a parity plot for the training and testing sets with additional
    marginal histograms.
    """
    limit = np.array(limit)
    
    plt.figure(figsize=(10, 10))
    gs = plt.GridSpec(20, 20)
    ax_main = plt.subplot(gs[4:, :16])
    ax_top = plt.subplot(gs[:4, :16])
    ax_right = plt.subplot(gs[4:, 16:])
    
    # Main scatter plot
    sns.scatterplot(x=df_train['Actual'], y=df_train['Predicted'], label='Train', alpha=0.9, s=50, ax=ax_main)
    sns.scatterplot(x=df_test['Actual'], y=df_test['Predicted'], label='Test', alpha=0.9, s=50, ax=ax_main)

    ax_main.plot(limit, limit, color='black', alpha=0.7, linewidth=2)
    ax_main.plot(limit, (1+error_line)*limit, color='black', linestyle='--', alpha=0.7, linewidth=2)
    ax_main.plot(limit, (1-error_line)*limit, color='black', linestyle='--', alpha=0.7, linewidth=2)

    if error_line == 0.01:
        ax_main.text(limit[1]-0.1, 1.01*limit[1]-0.18, '+1% error', verticalalignment='bottom', horizontalalignment='right', color='black', rotation=45)
        ax_main.text(limit[1]-0.05, 0.99*limit[1]-0.09, '-1% error', verticalalignment='top', horizontalalignment='right', color='black', rotation=45)

    elif error_line == 0.03:
        ax_main.text(limit[1]-0.2, 1.05*limit[1]-0.35, '+3% error', verticalalignment='bottom', horizontalalignment='right', color='black', rotation=45)
        ax_main.text(limit[1]-0.1, 0.95*limit[1]-0.04, '-3% error', verticalalignment='top', horizontalalignment='right', color='black', rotation=45)
    
    elif error_line == 0.05:
        ax_main.text(limit[1]-0.3, 1.05*limit[1]-0.38, '+5% error', verticalalignment='bottom', horizontalalignment='right', color='black', rotation=45)
        ax_main.text(limit[1]-0.2, 0.95*limit[1]-0.21, '-5% error', verticalalignment='top', horizontalalignment='right', color='black', rotation=45)
    
    ax_main.set_xlim(limit)
    ax_main.set_ylim(limit)
    ax_main.set_xlabel('True values')
    ax_main.set_ylabel('Predicted values')
    ax_main.legend(markerscale=1.5, loc='upper left', shadow=True, fancybox=True)

    # Marginal plot on the x-axis (top)
    sns.histplot(x=df_train['Actual'], label='Train', alpha=0.9, ax=ax_top)
    sns.histplot(x=df_test['Actual'], label='Test', alpha=0.8, ax=ax_top)
    
    ax_top.set_xlim(limit)
    ax_top.set_xlabel('')
    ax_top.set_ylabel('')
    ax_top.tick_params(axis='x', labelsize=0)
    ax_top.tick_params(axis='y', labelsize=0)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['top'].set_visible(False)
    ax_top.grid(False)
    
    plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 

    # Marginal plot on the y-axis (right)
    sns.histplot(y=df_train['Predicted'], label='Train', alpha=0.9, ax=ax_right)
    sns.histplot(y=df_test['Predicted'], label='Test', alpha=0.8, ax=ax_right)
    ax_right.set_ylim(limit)
    
    ax_right.set_xlabel('')
    ax_right.set_ylabel('')
    ax_right.tick_params(axis='x', labelsize=0)
    ax_right.tick_params(axis='y', labelsize=0)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['top'].set_visible(False)
    ax_right.grid(False)

    plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 

    filename = "Parity_plot_" + name + ".pdf"
    save_file(plt.gcf(), plots_path, filename, 'matplotlib', overwrite)
    plt.show()

def plot_residuals(df_train, df_test, name, plots_path, overwrite, limit, error_line, res_limit, mean):
    """
    Generate a residual plot for the training and testing sets.
    """
    plt.figure(figsize=(12, 10))
    gs = plt.GridSpec(20, 20)
    ax_main = plt.subplot(gs[4:, :16])
    ax_right = plt.subplot(gs[4:, 16:])
    
    # Main scatter plot
    sns.scatterplot(x=df_train['Actual'], y=df_train['Actual'] - df_train['Predicted'], label='Train', alpha=0.9, s=50, ax=ax_main)
    sns.scatterplot(x=df_test['Actual'], y=df_test['Actual'] - df_test['Predicted'], label='Test', alpha=0.9, s=50, ax=ax_main)
    
    ax_main.axhline(0, color='black', linestyle='-', alpha=0.7, linewidth=2)
    ax_main.axhline(error_line*mean, color='black', linestyle='--', alpha=0.7, linewidth=2)
    ax_main.axhline(-error_line*mean, color='black', linestyle='--', alpha=0.7, linewidth=2)
    
    if error_line == 0.01:
        ax_main.text(limit[1]-0.07, 0.0105*mean, '+1% error', verticalalignment='bottom', horizontalalignment='right', color='black')
        ax_main.text(limit[1]-0.07, -0.0105*mean, '-1% error', verticalalignment='top', horizontalalignment='right', color='black')
    
    elif error_line == 0.03:
        ax_main.text(limit[1]-0.07, 0.0305*mean, '+3% error', verticalalignment='bottom', horizontalalignment='right', color='black')
        ax_main.text(limit[1]-0.07, -0.0305*mean, '-3% error', verticalalignment='top', horizontalalignment='right', color='black')
    
    elif error_line == 0.05:
        ax_main.text(limit[1]-0.07, 0.0505*mean, '+5% error', verticalalignment='bottom', horizontalalignment='right', color='black')
        ax_main.text(limit[1]-0.07, -0.0505*mean, '-5% error', verticalalignment='top', horizontalalignment='right', color='black')
    
    ax_main.set_xlim(limit)
    ax_main.set_ylim(res_limit[0], res_limit[1])
    ax_main.set_xlabel('True values')
    ax_main.set_ylabel('Residuals')
    ax_main.legend(markerscale=1.5, loc='upper left', shadow=True, fancybox=True)
    
    # Marginal plot on the y-axis (right)
    sns.histplot(y=df_train['Actual'] - df_train['Predicted'], label='Train', alpha=0.9, ax=ax_right)
    sns.histplot(y=df_test['Actual'] - df_test['Predicted'], label='Test', alpha=0.8, ax=ax_right)
    
    ax_right.set_xlabel('')
    ax_right.set_ylabel('')
    ax_right.set_ylim(res_limit[0], res_limit[1])
    ax_right.tick_params(axis='x', labelsize=0)
    ax_right.tick_params(axis='y', labelsize=0)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['top'].set_visible(False)
    ax_right.grid(False)

    plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
  
    filename = "Residuals_plot_" + name + ".pdf"
    save_file(plt.gcf(), plots_path, filename, 'matplotlib', overwrite)
    plt.show()

def plot_residual_histogram(df_train, df_test, name, plots_path, overwrite, error_line, res_limit, mean):
    """
    Generate histograms of the residuals for the training and testing sets.
    """
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    yticks = ax.yaxis.get_major_ticks() 
    yticks[0].label1.set_visible(False)

    sns.histplot(df_train['Actual'] - df_train['Predicted'], label='Train', alpha=0.9)
    sns.histplot(df_test['Actual'] - df_test['Predicted'], label='Test', alpha=0.8)

    plt.axvline(0, color='black', linestyle='-', alpha=0.7, linewidth=2)
    plt.axvline(error_line*mean, color='black', linestyle='--', alpha=0.7, linewidth=2)
    plt.axvline(-error_line*mean, color='black', linestyle='--', alpha=0.7, linewidth=2)

    if error_line == 0.01:
        plt.text(0.011*mean, 400, '+1% error', verticalalignment='bottom', horizontalalignment='left', color='black', rotation=90)
        plt.text(-0.01*mean, 400, '-1% error', verticalalignment='bottom', horizontalalignment='right', color='black', rotation=90)

    elif error_line == 0.03:
        plt.text(0.032*mean, 400, '+3% error', verticalalignment='bottom', horizontalalignment='left', color='black', rotation=90)
        plt.text(-0.031*mean, 400, '-3% error', verticalalignment='bottom', horizontalalignment='right', color='black', rotation=90)

    elif error_line == 0.05:
        plt.text(0.052*mean, 400, '+5% error', verticalalignment='bottom', horizontalalignment='left', color='black', rotation=90)
        plt.text(-0.051*mean, 400, '-5% error', verticalalignment='bottom', horizontalalignment='right', color='black', rotation=90)

    plt.xlim(res_limit)
    plt.xlabel('True values')
    plt.ylabel('Residual')
    plt.legend(markerscale=1.5, loc='upper left', shadow=True, fancybox=True)
    
    filename = "ResidualsHistogram_plot_" + name + ".pdf"
    save_file(plt.gcf(), plots_path, filename, 'matplotlib', overwrite)
    plt.show()

def plot_distribution(df_train, df_test, name, plots_path, overwrite, limit, gridsize):
    """
    Generate smoothed kernel density estimate (KDE) plots for the actual and 
    predicted values.
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))

    sns.kdeplot(df_train['Actual'], linewidth=2, alpha=0.7, fill=True, ax=axs[0], label='Actual', gridsize=gridsize)
    sns.kdeplot(df_train['Predicted'], linewidth=2, alpha=0.7, fill=True, ax=axs[0], label='Predicted', gridsize=gridsize)
    axs[0].legend(markerscale=1.5, loc='upper left', shadow=True, fancybox=True)
    axs[0].set_xlabel('True values')
    axs[0].set_xlim(limit[0], limit[1])
    axs[0].set_title('Training')

    sns.kdeplot(df_test['Actual'], linewidth=2, alpha=0.7, fill=True, ax=axs[1], label='Actual', gridsize=gridsize)
    sns.kdeplot(df_test['Predicted'], linewidth=2, alpha=0.7, fill=True, ax=axs[1], label='Predicted', gridsize=gridsize)
    axs[1].legend(markerscale=1.5, loc='upper left', shadow=True, fancybox=True)
    axs[1].set_xlim(limit[0], limit[1])
    axs[1].set_xlabel('True values')
    axs[1].set_title('Testing')

    filename = "KDE_plot_" + name + ".pdf"
    save_file(plt.gcf(), plots_path, filename, 'matplotlib', overwrite)
    plt.show()

def plot_hisograms(df):
    """
    Function to generate histograms for the actual values and residuals using
    Freedman-Diaconis rule for bin width.
    """
    x = df['Actual']
    lower_percentile = np.percentile(x, 0.5)
    upper_percentile = np.percentile(x, 99.5)
    mask = (x > lower_percentile) & (x < upper_percentile)
    x_0_5 = x[mask.values].squeeze()
    x_0_5_res = x_0_5 - df['Predicted'][mask.values].squeeze()
    
    q25, q75 = np.percentile(x_0_5, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x_0_5) ** (-1/3)
    bins = round((x_0_5.max() - x_0_5.min()) / bin_width)
    print("Freedman-Diaconis number of bins:", bins)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    ax1.hist(x_0_5, bins=bins, edgecolor='black', alpha=0.8)
    ax1.set_xlabel('True values')
    ax1.set_ylabel('Frequency')

    ax2.hist(x_0_5_res, bins=bins, edgecolor='black', alpha=0.8)
    ax2.set_xlabel('Residual')
    ax2.set_ylabel('Frequency')

    plt.show()
    
def plot_mse_over_epochs(history, plots_path, overwrite):
    """
    Plots the mse over epochs for the training and testing sets.
    """
    plt.figure(figsize=(8, 6))  
    plt.plot(history.history['mean_squared_error'], label='Train MSE', linewidth=1)
    plt.plot(history.history['test_mse'], label='Test MSE', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend(markerscale=1.5, loc='upper left', shadow=True, fancybox=True)
    save_file(plt.gcf(), plots_path, "Loss_over_epoch.pdf", 'matplotlib', overwrite)
    plt.show()
     
def plot_time_predictions(df_train, df_test, plots_path, overwrite=False):
    fig, ax = plt.subplots(2, figsize=(16, 10))

    sns.scatterplot(ax=ax[0], x=df_train['Time'], y=df_train['Actual'], label='Actual', alpha=0.9, s=20)
    sns.scatterplot(ax=ax[0], x=df_train['Time'], y=df_train['Predicted'], label='Predicted', alpha=0.9, s=20)

    ax[0].set_title('Training', fontsize=20)
    ax[0].set_ylim(3.5, 4.5)
    ax[0].set_xlabel('Time', fontsize=15)
    ax[0].set_ylabel('SRD [ MJ/kg CO2]', fontsize=15)

    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%B %d'))
    ax[0].legend(markerscale=1.5)


    sns.scatterplot(ax=ax[1], x=df_test['Time'], y=df_test['Actual'], label='Actual', alpha=0.9, s=50)
    sns.scatterplot(ax=ax[1], x=df_test['Time'], y=df_test['Predicted'], label='Predicted', alpha=0.9, s=50)

    ax[1].set_title('Testing', fontsize=20)
    ax[1].set_xlabel('Time', fontsize=15)
    ax[1].set_ylabel('SRD [ MJ/kg CO2]', fontsize=15)

    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%B %d %H:00'))
    ax[1].legend(markerscale=1.5, loc='upper left', shadow=True, fancybox=True)

    plt.xticks(rotation=30, fontsize=15)
    plt.tight_layout()
    save_file(plt.gcf(), plots_path, "Predictions.pdf", 'matplotlib', overwrite)
    plt.show()
    
def generate_prediction_df(model, scaler, X, y):
    """
    Calculate the predictions for the model and inverse transform the data. Returns
    a DataFrame with the actual and predicted values in the original scale.
    """
    y_pred = model.predict(X.drop('Time', axis=1))
    y_pred = scaler.inverse_transform(y_pred)
    
    y_inv = y.copy()
    y_inv['SRD'] = scaler.inverse_transform(y['SRD'].to_numpy().reshape(-1, 1))

    df_res = pd.DataFrame({'Time': y_inv['Time'], 
                           'Actual': y_inv['SRD'], 
                           'Predicted': y_pred.flatten()})
    df_res.sort_values('Time')
    
    rmse = mean_squared_error(df_res['Actual'], df_res['Predicted'], squared=False)
    r2 = r2_score(df_res['Actual'], df_res['Predicted'])
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R2 Score: {r2}")
    
    return df_res, rmse, r2