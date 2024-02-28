import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import scipy.stats as stats
import seaborn as sns

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
    
def print_corr(df, col1, col2, method="pearson"):
    rho = df[col1].corr(df[col2], method=method)
    
    print("\nThe correlation between the following columns")
    print("---------------------------------------------")
    print(col1)
    print(col2)
    print("ρ =", np.round(rho, 2))
    print("---------------------------------------------")
    
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

def dropdown_plot(df):
    """
    Function to make a dropdown plot of the columns in the DataFrame
    """
    columns = df.columns[1:] # Columns without 'Time'

    # Create traces for each column
    traces = [go.Scatter(x=df['Time'], 
                         y=df[col], 
                         mode='markers', 
                         marker=dict(size=3), 
                         name=col)
              for col in columns]

    fig = go.Figure(data=traces)

    # Create a dropdown menu
    buttons = [dict(label=col,
                    method='update', 
                    args=[{'visible': [i == j for i in range(len(columns))]},
                          {'title': col}]
                    ) 
               for j, col in enumerate(columns)]
    updatemenu = dict(type='dropdown', 
                      x=0.5, 
                      y=1.15, 
                      showactive=True, 
                      buttons=buttons)
    fig.update_layout(updatemenus=[updatemenu])

    # Initially show only the first column
    fig.data[0].visible = True
    for trace in fig.data[1:]:
        trace.visible = False

    fig.show()
    
def qq_plot_all(dataframe, plots_per_row=7):
    """
    This function creates a matrix of QQ plots for all numeric columns in a pandas DataFrame using Matplotlib.
    There will be 'plots_per_row' plots on each row.
    """
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = len(numeric_columns)
    
    num_rows = int(np.ceil(num_cols / plots_per_row))
    
    fig, axes = plt.subplots(nrows=num_rows, ncols=plots_per_row, 
                             figsize=(plots_per_row * 5, num_rows * 5), 
                             constrained_layout=True)
    
    # Flatten axes array if more than one row
    if num_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Create QQ plots for each numeric column
    for i, column in enumerate(numeric_columns):
        stats.probplot(dataframe[column], dist="norm", plot=axes[i])
        axes[i].set_title(f'QQ plot for {column}')
        axes[i].get_lines()[0].set_color('royalblue')
        axes[i].get_lines()[1].set_color('salmon')  # Set the color of the reference line to red
    
    # Remove empty subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.show()
    
def generate_moisture_df(P_tot, T, P_H2O):
    Mm_CO2 = 44.01 # g/mol
    Mm_H2O = 18.02 # g/mol
    results = []
    for P in P_tot:
        # Convert P from barg to bar
        # Convert PH2O from MPa to bar
        x_CO2 = 1 - (P_H2O*10) / (P+1) # Mole fraction
        w_CO2 = x_CO2 * Mm_CO2 / (x_CO2 * Mm_CO2 + (1 - x_CO2) * Mm_H2O ) # Mass fraction
        results.append(w_CO2)

    moisture_df = pd.DataFrame(results, index=P_tot, columns=T)
    moisture_df = moisture_df.rename(columns=lambda x: str(x) + "°C").round(3)
    return moisture_df

def plot_TP_distribution(df):
    """
    Plot the distribution of P6 and T10 in a heatmap.

    Args:
        df (pandas.DataFrame): The input DataFrame containing all PFD data.

    Returns:
        None
    """
    
    # Picking colormap
    cmap_name = "viridis"
    cmap = cm.get_cmap(cmap_name)
    colormap_list = [cmap(i) for i in range(cmap.N)]

    # Create bins for P6 and T10
    p6_bins = np.linspace(0.4, 1.2, 6)
    t10_bins = np.linspace(10, 35, 11)

    # Create a binned DataFrame
    df_binned = pd.DataFrame({
        'P6': pd.cut(df['P6'], bins=p6_bins, include_lowest=True, right=False),
        'T10': pd.cut(df['T10'], bins=t10_bins, include_lowest=True, right=False)
    })

    # Calculate the percentage of data points in each bin
    heatmap_data = df_binned.groupby(['P6', 'T10']).size().reset_index(name='count')
    heatmap_data['percentage'] = 100 * heatmap_data['count'] / len(df)
    heatmap_pivot = heatmap_data.pivot("P6", "T10", "percentage")

    # Plotting the heatmap
    plt.figure(figsize=(16, 8), dpi=300)
    sns.set(font="Verdana")
    sns.set(rc={'axes.facecolor':colormap_list[0]})
    sns.heatmap(heatmap_pivot, annot=True, cmap=cmap_name, 
                fmt=".1f", cbar=True, mask=heatmap_pivot <= 0.15)
    plt.xlabel("T10 intervals [°C]", labelpad=15)
    plt.ylabel("P6 intervals [barg]", labelpad=15)
    plt.show()