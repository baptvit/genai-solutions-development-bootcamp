from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.tools import filter_restaurants_by_type_rating
from app.prompt import restaurant_system_prompt


def create_restaurant_agent_executor():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=1,
    )

    tools = [filter_restaurants_by_type_rating]

    llm_with_tools = llm.bind_tools(tools)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                restaurant_system_prompt,
            ),
            MessagesPlaceholder(variable_name="history"),
                        (
                "ai",
                "Hello there, Im the EPAM Bootcamp Restaurant Recommendation Assistant ! What's your name?  What type of cuisine are you looking for today?  And what is your preferred restaurant rating?",
            ),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
            "history": lambda x: x["history"],
        }
        | prompt
        # | prompt_trimmer # See comment above.
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )

    return AgentExecutor(agent=agent, tools=tools, verbose=False)
