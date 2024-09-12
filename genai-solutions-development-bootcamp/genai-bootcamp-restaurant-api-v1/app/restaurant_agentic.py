import os
import json
from typing import Any, List, Union

import pydantic

with open('/home/baptvit/genai-solutions-development-bootcamp/genai-bootcamp-restaurant-api/app/restaurant.json') as f:
    resultaurants = [json.loads(line) for line in f]
    #dict_list = [json.loads(line) for line in f]

import pandas as pd

df_restaurants = pd.DataFrame(resultaurants)

df_restaurants = df_restaurants[df_restaurants.rating != "Not yet rated"]

df_restaurants["rating"] = pd.to_numeric(df_restaurants["rating"])

from langchain_core.tools import tool


@tool
def filter_restaurants_by_type_rating(type_of_food: str, rating: float) -> str:
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
    return df_restaurants[(df_restaurants.type_of_food == type_of_food) & (df_restaurants.rating >= rating)].head(5).sort_values(by='rating', ascending=False).to_dict('records')
    #return "Restaurant flor de liz" 
tools = [filter_restaurants_by_type_rating]

os.environ["GOOGLE_API_KEY"] = ""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=1,
    max_tokens=5000,
    timeout=None,
)

#llm_with_tools = llm.bind_tools(tools)

from langchain_core.utils.function_calling import format_tool_to_openai_tool
llm_with_tools = llm.bind_tools(tools=[format_tool_to_openai_tool(tool) for tool in tools])

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

system_prompt = """You are a cheerful Restaurant Recommendation Assistant, here to help users find restaurants based on their preferences.
If a user asks questions unrelated to restaurant recommendations, politely let them know that your focus is on restaurant suggestions. If you are unable to answer a specific question, simply respond with: "Sorry, I can't help with that question. Make sure to use the filter_restaurants_by_type_rating tool for information."""

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        #("ai", "Hello there, Im the EPAM Bootcamp Restaurant Recommendation Assistant ! What's your name?  What type of cuisine are you looking for today?  And what is your preferred restaurant rating?"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage


# Construct the Tools agent
agent = create_tool_calling_agent(llm, tools, prompt)
# from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
# from langchain.agents.format_scratchpad.openai_tools import (
#     format_to_openai_tool_messages,
# )
# agent = (
#     {
#         "input": lambda x: x["input"],
#         "agent_scratchpad": lambda x: format_to_openai_tool_messages(
#             x["intermediate_steps"]
#         ),
#         "chat_history": lambda x: x["chat_history"],
#     }
#     | prompt
#     # | prompt_trimmer # See comment above.
#     | llm_with_tools
#     | OpenAIToolsAgentOutputParser()
# )

#agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
