import logging
from pathlib import Path
import pandas as pd
from typing import Tuple, Union
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


def split_csv_dataset(
    file_path: Union[str, Path],
    target_column_index: int,  # Now we expect an index, not a name
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a CSV dataset into training and validation sets using manual
    stratification
    """
    try:
        # Use header=None because the file has no header row.
        # Use float_precision='round_trip' for accuracy.
        df = pd.read_csv(file_path, header=None, float_precision='round_trip')
        logging.info(f"Successfully loaded {len(df)} rows from the CSV file.")
    except FileNotFoundError:
        logging.error(f"Error: The file '{file_path}' was not found.")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        logging.error(
            f"An unexpected error occurred : {e}"
            )
        return pd.DataFrame(), pd.DataFrame()
    # Check if the index is valid
    if target_column_index < 0 or target_column_index >= df.shape[1]:
        raise IndexError(
            f"Index {target_column_index} is out of bounds."
            )
    
    random.seed(seed)

    # Store the final training and validation dataframes
    train_df_list = []
    val_df_list = []

    target_column_data = df.iloc[:, target_column_index]
    unique_classes = target_column_data.unique()

    for cls in unique_classes:
        # Get all rows for the current class
        class_data = df[target_column_data == cls]
        # Shuffle the indices for this class
        shuffled_indices = list(class_data.index)
        random.shuffle(shuffled_indices)

        # Calculate the splite point
        num_val_samples = int(len(shuffled_indices) * val_ratio)

        # Split the indices
        val_indices = shuffled_indices[:num_val_samples]
        train_indices = shuffled_indices[num_val_samples:]

        # Append the split data to the lists
        train_df_list.append(df.loc[train_indices])
        val_df_list.append(df.loc[val_indices])

    # Concatenate the lists os dataaframes to form the final splits
    train_df = pd.concat(train_df_list).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_df_list).sample(frac=1, random_state=seed).reset_index(drop=True)

    # Rename the target column for clarity 
    train_df = train_df.rename(columns={train_df.columns[target_column_index]: 'diagnosis'})
    val_df = val_df.rename(columns={val_df.columns[target_column_index]: 'diagnosis'})
    return train_df, val_df


if __name__ == '__main__':
    CSV_FILE_PATH = Path("./data.csv")
    # The second column has an index of 1.
    TARGET_COLUMN_INDEX = 1
    try:
        train_data, val_data = split_csv_dataset(
            file_path=CSV_FILE_PATH,
            target_column_index=TARGET_COLUMN_INDEX,
            val_ratio=0.2,
            seed=42
        )

        if not train_data.empty and not val_data.empty:
            logging.info("\nSplit successful! ✨")
            logging.info(f"Training set size: {len(train_data)} rows")
            logging.info(f"Validation set size: {len(val_data)} rows")

            logging.info("\nClass distribution in training set:")
            # Use iloc to acess the target columns by its new name after rename
            logging.info(train_data['diagnosis'].value_counts(normalize=True))
            logging.info("\nClass distribution in validation set:")
            logging.info(val_data['diagnosis'].value_counts(normalize=True))

            train_data.to_csv("train_split.csv", index=False)
            val_data.to_csv("val_split.csv", index=False)
            logging.info(
                "\nSplits saved to 'train_split.csv' and 'val_split.csv'.")
    except (ValueError, IndexError) as e:
        logging.error(f"Error during split: {e}")
