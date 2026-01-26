import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")
    train_data_path: str = os.path.join("artifacts", "train_data.csv")
    test_data_path: str = os.path.join("artifacts", "test_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process.")
        try:
            #source_file_path = "notebook/data/stud.csv" 
            # Read the dataset
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info(f"Data read to Dataframe.")

            # Ensure the artifacts directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}.")

            # Split the data into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Data split into training and testing sets.")

            # Save the training and testing data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Training data saved at {self.ingestion_config.train_data_path}.")
            logging.info(f"Testing data saved at {self.ingestion_config.test_data_path}.")

            logging.info("Data ingestion process completed successfully.")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logging.info("Error occurred during data ingestion.")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        obj = DataIngestion()
        obj.initiate_data_ingestion("notebook/data/stud.csv")
    except Exception:
        logging.exception("Data ingestion failed at entry point")
        raise
