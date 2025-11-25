
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")

class PlotUtils:
    """
    A utility class for plotting data in the Loan Prediction project.
    Supports histogram, scatter, line plots, and prediction comparison.
    """

    def __init__(self, df):
        self.df = df

    def histogram(self, column, bins=10, title=None, save_path=None):
        plt.figure(figsize=(8, 5))
        sns.histplot(self.df[column], bins=bins, kde=True)
        plt.title(title or f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def scatter(self, x_col, y_col, hue=None, title=None, save_path=None):
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=self.df, x=x_col, y=y_col, hue=hue)
        plt.title(title or f'Scatter plot of {y_col} vs {x_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def line(self, x_col, y_col, title=None, save_path=None):
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=self.df, x=x_col, y=y_col)
        plt.title(title or f'Line plot of {y_col} vs {x_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def boxplot(self, column, by=None, title=None, save_path=None):
        
        plt.figure(figsize=(8, 5))
        
        if by:
            sns.boxplot(data=self.df, x=by, y=column)
            plt.xlabel(by)
            plt.ylabel(column)
            plt.title(title or f'Box plot of {column} grouped by {by}')
        else:
            sns.boxplot(y=self.df[column])
            plt.ylabel(column)
            plt.title(title or f'Box plot of {column}')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()
 