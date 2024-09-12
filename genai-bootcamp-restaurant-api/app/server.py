import pydantic
import re

from typing import Any, List, Union
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from langserve import add_routes
from pathlib import Path
from typing import Callable
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory


from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage

from app.agent import create_restaurant_agent_executor

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


class Input(pydantic.BaseModel):
    input: str = pydantic.Field(
        ...,
        max_length=5000,
        description="The human input to the chat system.",
        extra={"widget": {"type": "chat", "input": "input"}},
    )
    # The field extra defines a chat widget.
    # Please see documentation about widgets in the main README.
    # The widget is used in the playground.
    # Keep in mind that playground support for agents is not great at the moment.
    # To get a better experience, you'll need to customize the streaming output
    # for now.
    chat_history: List[Union[HumanMessage, AIMessage, FunctionMessage]] = (
        pydantic.Field(
            ...,
            extra={"widget": {"type": "chat", "input": "input", "output": "output"}},
        )
    )


class Output(pydantic.BaseModel):
    output: Any


def _is_valid_identifier(value: str) -> bool:
    """Check if the session ID is in a valid format."""
    # Use a regular expression to match the allowed characters
    valid_characters = re.compile(r"^[a-zA-Z0-9-_]+$")
    return bool(valid_characters.match(value))


def create_session_factory(
    base_dir: Union[str, Path],
) -> Callable[[str], BaseChatMessageHistory]:
    """Create a session ID factory that creates session IDs from a base dir.

    Args:
        base_dir: Base directory to use for storing the chat histories.

    Returns:
        A session ID factory that creates session IDs from a base path.
    """
    base_dir_ = Path(base_dir) if isinstance(base_dir, str) else base_dir
    if not base_dir_.exists():
        base_dir_.mkdir(parents=True)

    def get_chat_history(session_id: str) -> FileChatMessageHistory:
        """Get a chat history from a session ID."""
        if not _is_valid_identifier(session_id):
            raise HTTPException(
                status_code=400,
                detail=f"Session ID `{session_id}` is not in a valid format. "
                "Session ID must only contain alphanumeric characters, "
                "hyphens, and underscores.",
            )
        file_path = base_dir_ / f"{session_id}.json"
        return FileChatMessageHistory(str(file_path))

    return get_chat_history


chain_with_history = (
    RunnableWithMessageHistory(
        create_restaurant_agent_executor(),
        create_session_factory("chat_histories"),
        input_messages_key="input",
        history_messages_key="history",
    )
    .with_types(input_type=Input, output_type=Output)
    .with_config({"run_name": "agent"})
)

# Edit this to add the chain you want to add
add_routes(app, chain_with_history, path="/restaurant", playground_type="default")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
