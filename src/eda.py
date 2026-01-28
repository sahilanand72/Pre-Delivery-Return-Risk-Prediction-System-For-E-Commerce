import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_return_distribution(df: pd.DataFrame):
    sns.countplot(x='returned', data=df)
    plt.title("Return vs Not Returned")
    plt.show()


def plot_price_vs_return(df: pd.DataFrame):
    sns.boxplot(x='returned', y='price', data=df)
    plt.title("Product Price vs Return")
    plt.show()


def plot_discount_vs_return(df: pd.DataFrame):
    sns.boxplot(x='returned', y='discount_percentage', data=df)
    plt.title("Discount Percentage vs Return")
    plt.show()


def plot_category_return_rate(df: pd.DataFrame):
    return_rate = (
        df.groupby('product_category')['returned']
        .mean()
        .sort_values(ascending=False)
    )

    return_rate.plot(kind='bar', figsize=(10, 4))
    plt.title("Return Rate by Product Category")
    plt.ylabel("Return Rate")
    plt.show()
