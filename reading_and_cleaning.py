import pandas as pd
import seaborn as seaborn
from rapidfuzz import process, fuzz

DATASET_PATH = "../vehicles.csv"

conditionMap = {
    "new": 1,
    "like new": 2,
    "excellent": 3,
    "good": 4,
    "fair": 5,
    "salvage": 6
}

title_statusMap = {
    "clean": 1,
    "salvage": 2,
    "rebuilt": 3,
    "lien": 4,
    "missing": 5,
    "parts only": 6
}


def read_data():
    dataset = pd.read_csv(DATASET_PATH)
    
    return dataset
    
def clean_data(dataset: pd.DataFrame):
    
    #Extracting number of the cylinders
    dataset["cylinders"] = dataset["cylinders"].str.extract(r'(\d+)')
    dataset["cylinders"] = pd.to_numeric(dataset["cylinders"], downcast="unsigned", errors="coerce")
    
    #Converting condition to numeric values
    dataset["condition"] = dataset["condition"].map(conditionMap)
    
    #Converting title_status to numeric values
    dataset["title_status"] = dataset["title_status"].map(title_statusMap)
    
    #Check if VIN present
    dataset["vin_present"] = dataset["VIN"].notnull().astype(bool)
    
    #Drop county
    dataset.drop(columns=["county"], inplace=True)
    
    #Drop where price is null
    dataset.dropna(subset=["price"], inplace=True)
    dataset = dataset[dataset["price"] > 0]
    
    dataset["condition"] = dataset["condition"].fillna("unknown")
    dataset["fuel"] = dataset["fuel"].fillna("unknown")
    dataset["transmission"] = dataset["transmission"].fillna("unknown")
    
    # Convert types to numberics
    dataset["odometer"] = pd.to_numeric(dataset["odometer"], downcast="unsigned", errors="coerce")
    dataset["year"] = pd.to_numeric(dataset["year"], downcast="unsigned", errors="coerce")
    dataset["price"] = pd.to_numeric(dataset["price"], downcast="integer", errors="coerce")
    
    #dataset["posting_date"] = pd.to_datetime(dataset["posting_date"])
    dataset = dataset[(dataset["price"] < 100000) & (dataset["price"] > 500)]
    dataset = dataset[(dataset["year"] > 1980) & (dataset["year"] <= 2026)]
    
    dataset["model"] = dataset["model"].str.lower().str.strip().fillna("unknown")
    dataset["manufacturer"] = dataset["manufacturer"].str.lower().str.strip().fillna("unknown")
    
    return dataset

def read_car_types():
    car_types = pd.read_csv("../Car Dataset 1945-2020.csv")
    
    car_types["Modle"] = car_types["Modle"].str.lower().str.strip()
    car_types["Make"] = car_types["Make"].str.lower().str.strip()
    
    car_model_dicts = car_types.groupby("Make")["Modle"].apply(lambda x: sorted(x.unique())).to_dict()
    
    return car_model_dicts

def match_new_model(manufacturer: str, given_model: str, car_model_dicts: dict, confidence: int):
    choices = car_model_dicts.get(manufacturer, [])
    
    if not choices:
        return "unknown"
    
    text = str(given_model).lower().strip()
    tokens = text.split()

    candidates = [
        text,
        " ".join(tokens[:3]),
        " ".join(tokens[:2]),
        tokens[0]
    ]

    for candidate in candidates:
        if not candidate:
            continue
            
        match = process.extractOne(
            candidate,
            choices,
            scorer=fuzz.token_sort_ratio
        )
        
        if match and match[1] >= confidence:
            return match[0]

    return "unknown"
    
    

def main():
    dataset = read_data()
    dataset = clean_data(dataset)

    car_types_dict = read_car_types()
    
    dataset["model_clean"] = dataset.apply(
        lambda row: match_new_model(
            row["manufacturer"],
            row["model"],
            car_types_dict,
            confidence=70
        ),
        axis = 1
    )
    
    dataset.to_csv("../cleaned_vehicles.csv", index=False)

if __name__ == '__main__':
    main()