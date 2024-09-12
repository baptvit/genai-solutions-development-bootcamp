import json
import pandas as pd


def restaurant_data(
    path: str = "/home/baptvit/genai-bootcamp-restaurant-api/app/data/restaurant.json",
) -> pd.DataFrame:
    with open(path) as f:
        resultaurants = [json.loads(line) for line in f]

    df_restaurants = pd.DataFrame(resultaurants)

    df_restaurants = df_restaurants[df_restaurants.rating != "Not yet rated"]

    df_restaurants["rating"] = pd.to_numeric(df_restaurants["rating"])

    df_restaurants = df_restaurants.drop("_id", axis=1)

    return df_restaurants.drop_duplicates()
