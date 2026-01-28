from src.data_loader import load_data
from src.preprocessing import (
    create_target,
    remove_leakage,
    handle_missing_values,
    encode_categorical_features
)
from src.eda import (
    plot_return_distribution,
    plot_price_vs_return,
    plot_discount_vs_return,
    plot_category_return_rate
)

from sklearn.model_selection import train_test_split


DATA_PATH = "data/ecommerce_returns_synthetic_data.csv"


def main():
    # Load data
    df = load_data(DATA_PATH)
    print(df.shape)
    print(df.info())

    # Create target
    df = create_target(df)

    # Remove leakage
    df = remove_leakage(df)

    # Handle missing values
    df = handle_missing_values(df)

    # EDA
    plot_return_distribution(df)
    plot_price_vs_return(df)
    plot_discount_vs_return(df)
    plot_category_return_rate(df)

    # Prepare features & target
    X = df.drop(columns=['returned'])
    y = df['returned']

    print(X.shape, y.shape)

    # Encode categoricals
    X, label_encoders = encode_categorical_features(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)


if __name__ == "__main__":
    main()
