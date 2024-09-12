from typing import List
from langchain_core.tools import tool

from app.ingestion import restaurant_data


NUM_RECOMENDATIONS = 10

@tool
def filter_restaurants_by_type_rating(type_of_food: str, rating: float) -> List[dict]:
    """Filter restaurants by the food type and the restaurant rating.

    Args:
        type_of_food: Type of food served in the restaurant Examples: 'Chinese', 'Thai', 'Curry', 'Pizza', 'African', 'Turkish',
       'American', 'Fish & Chips', 'English', 'Chicken', 'Arabic',
       'Desserts', 'Kebab', 'Portuguese', 'Persian', 'Peri Peri',
       'South Curry', 'Lebanese', 'Punjabi', 'Caribbean', 'Ethiopian',
       'Greek', 'Moroccan', 'Russian', 'Afghan', 'Cakes', 'Sushi',
       'Japanese', 'Burgers', 'Grill', 'Bangladeshi', 'Middle Eastern',
       'Mexican', 'Mediterranean', 'Pasta', 'Sri-lankan', 'Breakfast',
       'Pakistani', 'Sandwiches', 'Vegetarian', 'Vietnamese', 'Polish',
       'Azerbaijan', 'Milkshakes', 'Spanish', 'Nigerian', 'Ice Cream',
       'Korean', 'Bagels', 'Pick n Mix' and 'Healthy'.
        rating: Avaluation rate for each restaurant (0.0 to 5.0)
    """
    df_restaurants = restaurant_data()
    return (
        df_restaurants[
            (df_restaurants.type_of_food == type_of_food)
            & (df_restaurants.rating >= rating)
        ]
        .head(NUM_RECOMENDATIONS)
        .sort_values(by="rating", ascending=False)
        .to_dict("records")
    )
