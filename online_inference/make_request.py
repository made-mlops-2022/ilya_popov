import pandas as pd
import requests


if __name__ == "__main__":
    data = pd.read_csv("data/raw/heart_cleveland_upload.csv")
    data.drop(columns="condition", inplace=True)
    feature_names = list(data.columns)
    for i in range(10):
        request_data = data.iloc[0].tolist()
        print(request_data)
        response = requests.get(
            "http://127.0.0.1:8000/predict/",
            json={"data": [request_data], "feature_names": feature_names}
        )
        print(response.status_code)
        print(response.json())
