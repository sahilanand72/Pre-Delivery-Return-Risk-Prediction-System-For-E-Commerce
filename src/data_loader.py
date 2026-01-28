import pandas as pd
from pathlib import Path


def load_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load the e-commerce return dataset from a CSV file.

    Parameters
    ----------
    file_path : str or Path
        Relative or absolute path to the dataset CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset as a pandas DataFrame.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at path: {file_path}")

    df = pd.read_csv(file_path)

    return df
