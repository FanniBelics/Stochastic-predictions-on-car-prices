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

luxury_brands = ["rolls-royce", "bentley", "lamborghini", "ferrari", "mclaren",\
                "aston martin", "bugatti", "maserati", "porsche", "jaguar",\
                    "cadillac", "infiniti", "acura", "lincoln"]


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
    dataset["vin_present"] = dataset["VIN"].notnull().astype(int)
    
    #Drop county
    dataset.drop(columns=["county"], inplace=True)
    
    #Drop where price is null
    dataset.dropna(subset=["price"], inplace=True)
    dataset = dataset[dataset["price"] > 0]
    
    dataset["condition"] = dataset["condition"].fillna(0).astype(int)
    dataset["title_status"] = dataset["title_status"].fillna(0).astype(int)
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
    dataset["manufacturer"] = dataset["manufacturer"].str.lower().str.strip().dropna()
    dataset["is_manufacturer_unknown"] = (dataset["manufacturer"] == "unknown").astype(int)
    
    dataset["car_age"] = 2026 - dataset["year"]
    
    dataset = dataset[dataset["odometer"] >= 0]
    dataset["mileage_per_year"] = dataset["odometer"] / dataset["car_age"]
    dataset["mileage_per_year"] = dataset["mileage_per_year"].fillna(0)
    dataset = dataset[dataset["mileage_per_year"] < 80000]
    
    dataset["luxury_brand"] = dataset["manufacturer"].apply(lambda x: 1 if x in luxury_brands else 0)
    
    dataset.drop(columns=["VIN"], inplace=True)
    dataset.drop(columns=["url"], inplace=True)
    dataset.drop(columns=["region_url"], inplace=True)
    dataset.drop(columns=["lat"], inplace=True)
    dataset.drop(columns=["long"], inplace=True)
    dataset.drop(columns=["model"], inplace=True)    
    
    return dataset

def read_car_types():
    car_types = pd.read_csv("../Car Dataset 1945-2020.csv")
    
    car_types["Modle"] = car_types["Modle"].str.lower().str.strip()
    car_types["Make"] = car_types["Make"].str.lower().str.strip()
    
    car_model_dicts = car_types.groupby("Make")["Modle"].apply(lambda x: sorted(x.unique())).to_dict()
    
    return (car_model_dicts, car_types)

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
    
def find_cylinder_number(dataset: pd.DataFrame, car_types_dataset: pd.DataFrame):
    car_types_dataset = car_types_dataset.rename(columns={"Make": "manufacturer", "Modle": "model_clean", "number_of_cylinders": "cylinders"})
    car_types_dataset = car_types_dataset[["manufacturer", "model_clean", "cylinders"]].drop_duplicates()
    
    lookup = (
    car_types_dataset[["manufacturer","model_clean","cylinders"]]
    .drop_duplicates()
    .set_index(["manufacturer","model_clean"])["cylinders"]
    .to_dict()
    )   

    keys = pd.Series(list(zip(dataset["manufacturer"], dataset["model_clean"])))
    dataset["cylinders_fill"] = keys.map(lookup)

    dataset["cylinders"] = dataset["cylinders"].fillna(dataset["cylinders_fill"])
    dataset.drop(columns=["cylinders_fill"], inplace=True)
        
    return dataset

def main():
    dataset = read_data()
    car_types_dict, car_types = read_car_types()
    
    dataset["model_clean"] = dataset.apply(
        lambda row: match_new_model(
            row["manufacturer"],
            row["model"],
            car_types_dict,
            confidence=70
        ),
        axis = 1
    )
    
    dataset = clean_data(dataset)
    dataset = find_cylinder_number(dataset, car_types)
    
    dataset[["price","car_age","odometer","mileage_per_year","cylinders","luxury_brand"]].corr()
    
    dataset.to_csv("../cleaned_vehicles.csv", index=False)

if __name__ == '__main__':
    main()