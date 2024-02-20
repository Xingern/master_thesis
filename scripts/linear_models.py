import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real

SAVE_FIG = True


def BayesSearchCV_results(X_test, y_test, bs, filename):
    # Print the results
    print(f"Best hyperparameters: {bs.best_params_}")
    print("--------------------------------------------")
    print(f"Best RMSE on the left out data: {-bs.best_score_}")
    print("Best RMSE on test:", -bs.score(X_test.drop("Time", axis=1), y_test)) 
    
    # Get the RMSE scores across each hyperparameter combination
    results = bs.cv_results_
    mean_test_scores = -results['mean_test_score']
    if len(bs.best_params_.keys()) == 1:
        alpha = list(bs.best_params_.keys())[0]
        best_alpha = bs.best_params_[alpha]
        alpha_values = np.array([params[alpha] for params in results['params']])
        
    elif len(bs.best_params_.keys()) == 2:
        def calc_L1_L2(alpha, l1_ratio):
            return alpha * l1_ratio, alpha * (1 - l1_ratio)
        
        alpha, l1 = list(bs.best_params_.keys())
        a, b = bs.best_params_.values()
        best_l1, best_alpha = calc_L1_L2(a, b)
        alphas = np.array([params[alpha] for params in results['params']])
        l1_ratios = np.array([params[l1] for params in results['params']])
        alphas, l1_ratios = calc_L1_L2(np.array(alphas), np.array(l1_ratios))
        mask = (l1_ratios != 0)
        l1_ratios = l1_ratios[mask]
    
    # Make the plot
    props = dict(boxstyle='round', facecolor='wheat', alpha=1)
    plt.figure(figsize=(5, 3), dpi=200)
    if 'lasso' in alpha:
        indexs_to_order_by = alpha_values.argsort()
        plt.plot(alpha_values[indexs_to_order_by], 
                 mean_test_scores[indexs_to_order_by], 
                 marker='o')
        plt.xlabel('L1 regularization parameter')
        plt.text(0.25, 0.85, 
            "Optimal L1 term: {:.3f}".format(best_alpha), 
            fontsize=12,
            horizontalalignment='center',
            verticalalignment='center', 
            transform=plt.gca().transAxes, 
            bbox=props)
        plt.xscale('log')
        
    elif 'ridge' in alpha:  
        indexs_to_order_by = alpha_values.argsort()
        plt.plot(alpha_values[indexs_to_order_by], 
                 mean_test_scores[indexs_to_order_by], 
                 marker='o')
        plt.xlabel('L2 regularization parameter')
        plt.text(0.25, 0.85, 
                "Optimal L2 term: {:.3f}".format(best_alpha), 
                fontsize=12,
                horizontalalignment='center',
                verticalalignment='center', 
                transform=plt.gca().transAxes, 
                bbox=props)
        plt.xscale('log')
        
    elif 'elastic' in alpha:
        indexs_to_order_by = l1_ratios.argsort()
        plt.plot(l1_ratios[indexs_to_order_by], 
                 mean_test_scores[indexs_to_order_by], 
                 marker='o', 
                 label='L1')
        indexs_to_order_by = alphas.argsort()
        plt.plot(alphas[indexs_to_order_by], 
                mean_test_scores[indexs_to_order_by], 
                marker='o', 
                label='L2')
        plt.xlabel('Regularization parameter')
        plt.text(0.25, 0.85, 
         "Optimal L1 term: {:.3f}\nOptimal L2 term: {:.3f}".format(best_l1, best_alpha), 
         fontsize=12,
         horizontalalignment='center',
         verticalalignment='center', 
         transform=plt.gca().transAxes, 
         bbox=props)
        plt.legend(loc='lower right')
        plt.xscale('log')
        
    elif 'pls' in alpha:
        indexs_to_order_by = alpha_values.argsort()
        plt.plot(alpha_values[indexs_to_order_by], 
                 mean_test_scores[indexs_to_order_by], 
                 marker='o')
        plt.xticks(range(int(min(alpha_values)), int(max(alpha_values))+1))
        plt.xlabel('Number of principal components')
        
    plt.ylabel('RMSE')
    if SAVE_FIG:
        plt.savefig("../figs/linear_models/" + filename, bbox_inches='tight')
    plt.show()
     
     
def split_data(df, method):
    X = df.drop(['SRD'], axis=1)
    y = df['SRD']

    if method == "time-series":
        date = "2020-06-15 00:00"
        X_train = df[df["Time"] < date].drop(['SRD'], axis=1)
        X_test = df[df["Time"] > date].drop(['SRD'], axis=1)
        y_train = df[df["Time"] < date]["SRD"]
        y_test = df[df["Time"] > date]["SRD"]

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=0.20, 
                                                            random_state=0,
                                                            shuffle=True)

    return X_train, X_test, y_train, y_test


def SFS_results(X_train, X_test, y_test, bs, filename):
    best_variables = bs.best_estimator_.named_steps['sfs'].k_feature_names_
    total_variables = len(X_train.columns) - 1
    print(f"{len(best_variables)} out of {total_variables} variables were selected during SFS")
    print(f"Best variables: {best_variables}")
    print(f"Best hyperparameters: {bs.best_params_}")
    print("--------------------------------------------")
    print(f"Best RMSE on the left out data: {-bs.best_score_}")
    print("Best RMSE on test:", -bs.score(X_test.drop("Time", axis=1), y_test))

    # Get the RMSE scores across each SFS combination
    avg_scores = []
    for subset_size, subset_info in bs.best_estimator_.named_steps['sfs'].subsets_.items():
        avg_scores.append(np.sqrt(-subset_info["avg_score"]))
        
    plt.figure(figsize=(5, 3), dpi=200)
    plt.plot([i for i in range(1, len(avg_scores)+1)], avg_scores, marker='o')
    plt.xlabel('Number of Variables')
    plt.ylabel('RMSE')
    plt.xticks([i for i in range(1, len(avg_scores)+1)])
    if SAVE_FIG:
        plt.savefig("../figs/linear_models/" + filename, bbox_inches='tight')
    plt.show()
  

def plot_training_test(X_train, y_train, X_test, y_test):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=X_train["Time"], 
            y=y_train.squeeze(), 
            mode='markers', 
            marker=dict(size=4),
            name='Train',
        )
    )

    fig.add_trace(
        go.Scatter(
            x=X_test["Time"], 
            y=y_test.squeeze(), 
            mode='markers', 
            marker=dict(size=4),
            name='Test',
        )
    )

    fig.update_layout(title="Train and Test Data", 
                    legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=0.15
                        ),
                    xaxis_title="Time", 
                    yaxis_title="SRD",
                    template="seaborn")

    fig.show()


def plot_parity_residual(X_test, y_test, bs, filename, rng=[2, 6.5]):
    y_pred = bs.predict(X_test.drop("Time", axis=1)).squeeze()
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), dpi=200)
    plt.style.use('seaborn')
    
    # Scatter plot
    axs[0, 0].scatter(y_test, y_pred)
    axs[0, 0].plot([0, 34], [0, 34], color='red')
    axs[0, 0].set_xlabel("True values")
    axs[0, 0].set_ylabel("Predicted values")
    axs[0, 0].set_title("Parity plot")

    # Residuals plot
    residuals = y_test - y_pred
    axs[0, 1].scatter(y_pred, residuals)
    axs[0, 1].axhline(y=0, color='red', linestyle='--')
    axs[0, 1].set_xlabel("Predicted values")
    axs[0, 1].set_ylabel("Residuals")
    axs[0, 1].set_title("Residuals plot")

    lower_percentile = np.percentile(y_test, 0.5)
    upper_percentile = np.percentile(y_test, 99.5)
    mask = (y_test > lower_percentile) & (y_test < upper_percentile)

    y_test_0_5 = y_test[mask.values].squeeze()
    y_pred_0_5 = y_pred[mask.values]

    # Scatter plot
    axs[1, 0].scatter(y_test_0_5, y_pred_0_5)
    axs[1, 0].plot(rng, rng, color='red')
    axs[1, 0].set_xlim(rng)
    axs[1, 0].set_ylim(rng)
    axs[1, 0].set_xlabel("True values")
    axs[1, 0].set_ylabel("Predicted values")
    axs[1, 0].set_title("Exluding 0.5th and 99.5th percentiles")

    # Residuals plot
    residuals = y_test_0_5 - y_pred_0_5
    axs[1, 1].scatter(y_pred_0_5, residuals)
    axs[1, 1].axhline(y=0, color='red', linestyle='--')
    axs[1, 1].set_xlabel("Predicted values")
    axs[1, 1].set_ylabel("Residuals")
    axs[1, 1].set_title("Exluding 0.5th and 99.5th percentiles")

    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig("../figs/linear_models/" + filename, bbox_inches='tight')
    plt.show()


def plot_prediction(df, X_train, X_test, bs):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Time"], 
            y=df["SRD"], 
            mode='markers', 
            marker=dict(size=4),
            name='Actual',
        )
    )

    fig.add_trace(
        go.Scatter(
            x=X_train["Time"], 
            y=bs.predict(X_train.drop("Time", axis=1)), 
            mode='markers', 
            marker=dict(size=4),
            name='Training',
        )
    )

    fig.add_trace(
        go.Scatter(
            x=X_test["Time"], 
            y=bs.predict(X_test.drop("Time", axis=1)), 
            mode='markers', 
            marker=dict(size=4),
            name='Testing',
        )
    )

    fig.update_layout(title="Modelling the SRD using linear regression", 
                    legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=0.3
                        ),
                    template="seaborn",
                    xaxis_title="Time", 
                    yaxis_title="Specific Reboiler Duty [MJ/kgCO2]")

    fig.show()
    
    
def plot_coefficients(bs, named_step, filename):
    if 'ols' in named_step:
        feature_names = bs.best_estimator_.named_steps['sfs'].k_feature_names_
    else:
        feature_names = bs.best_estimator_.named_steps[named_step].feature_names_in_
    
    feature_coef = bs.best_estimator_.named_steps[named_step].coef_
    
    df = pd.DataFrame({'Feature': feature_names,
                       'Coefficient': feature_coef})

    fig = go.Figure(data=[go.Bar(x=df['Feature'], y=df['Coefficient'])])  
    
    fig.update_layout(
                    xaxis_title="Feature",
                    yaxis_title="Coefficient",
                    height=400,
                    width=800,
                    legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=0.15
                            ),
                    template="seaborn",
                    margin=dict(l=0, r=20, t=20, b=10),
                    )
    
    fig.show()
    if SAVE_FIG:
        fig.write_image("../figs/linear_models/" + filename)
    
    
def plot_PLS_coefficients(gs, X_train, filename, n_components=1):
    x_scores = gs.best_estimator_.named_steps['plsr'].x_scores_
    x_loadings = gs.best_estimator_.named_steps['plsr'].x_loadings_
    n = x_loadings.shape[0]
    variable_names = X_train.columns.values[1:]
    n_features = len(variable_names)
    loadings = x_loadings

    if n_components == 1:
        fig, axes = plt.subplots(1, n_components, figsize=(8, 4), dpi=200)
        axes = [axes, 0]
    else:
        fig, axes = plt.subplots(1, n_components, figsize=(12, 4), dpi=200)
        
    for i in range(n_components):
        axes[i].bar(range(n_features), loadings[:, i])
        axes[i].set_title(f'Loadings for PC{i+1}')
        axes[i].set_xlabel('Variable')
        axes[i].set_ylabel('Loading Value')
        axes[i].set_xticks(range(n_features))
        axes[i].set_xticklabels(variable_names, rotation=45, ha="right")

    if SAVE_FIG:
        plt.savefig("../figs/linear_models/" + filename, bbox_inches='tight')
    plt.show()