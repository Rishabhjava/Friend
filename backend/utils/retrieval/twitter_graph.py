import datetime
import uuid
import asyncio
from typing import List, Optional, Tuple, AsyncGenerator

from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

import database.notifications as notification_db
from models.chat import ChatSession, Message
from models.plugin import Plugin
from utils.llm import (
    qa_rag_with_twitter,
    qa_rag_with_twitter_stream,
    extract_question_from_conversation,
)
from utils.other.endpoints import timeit


class AsyncStreamingCallback(BaseCallbackHandler):
    def __init__(self):
        self.queue = asyncio.Queue()

    async def put_data(self, text):
        await self.queue.put(f"data: {text}")

    async def put_thought(self, text):
        await self.queue.put(f"think: {text}")

    def put_thought_nowait(self, text):
        self.queue.put_nowait(f"think: {text}")

    async def end(self):
        await self.queue.put(None)

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        await self.put_data(token)

    async def on_llm_end(self, response, **kwargs) -> None:
        await self.end()

    async def on_llm_error(self, error: Exception, **kwargs) -> None:
        print(f"Error on LLM {error}")
        await self.end()

    def put_data_nowait(self, text):
        self.queue.put_nowait(f"data: {text}")

    def end_nowait(self):
        self.queue.put_nowait(None)


class TwitterGraphState(TypedDict):
    uid: str
    messages: List[Message]
    plugin_selected: Optional[Plugin]
    tz: str
    cited: Optional[bool] = False
    streaming: Optional[bool] = False
    callback: Optional[AsyncStreamingCallback] = None
    twitter_context: Optional[str] = None
    parsed_question: Optional[str]
    answer: Optional[str]
    ask_for_nps: Optional[bool]


def determine_conversation(state: TwitterGraphState):
    question = extract_question_from_conversation(state.get("messages", []))
    print("determine_conversation parsed question:", question)
    return {"parsed_question": question}


def qa_handler(state: TwitterGraphState):
    uid = state.get("uid")
    twitter_context = state.get("twitter_context", "")

    # streaming
    streaming = state.get("streaming")
    if streaming:
        response: str = qa_rag_with_twitter_stream(
            uid,
            state.get("parsed_question"),
            "",  # No memories context for Twitter
            twitter_context,
            state.get("plugin_selected"),
            cited=state.get("cited"),
            messages=state.get("messages"),
            tz=state.get("tz"),
            callbacks=[state.get('callback')]
        )
        return {"answer": response, "ask_for_nps": True}

    # no streaming
    response: str = qa_rag_with_twitter(
        uid,
        state.get("parsed_question"),
        "",  # No memories context for Twitter
        twitter_context,
        state.get("plugin_selected"),
        cited=state.get("cited"),
        messages=state.get("messages"),
        tz=state.get("tz"),
    )
    return {"answer": response, "ask_for_nps": True}


# Create and configure the workflow
workflow = StateGraph(TwitterGraphState)

# Add nodes and edges
workflow.add_edge(START, "determine_conversation")
workflow.add_node("determine_conversation", determine_conversation)
workflow.add_edge("determine_conversation", "qa_handler")
workflow.add_node("qa_handler", qa_handler)
workflow.add_edge("qa_handler", END)

# Compile graphs
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
graph_stream = workflow.compile()


@timeit
def execute_twitter_graph_chat(
        uid: str, messages: List[Message], twitter_context: str,
        plugin: Optional[Plugin] = None, cited: Optional[bool] = False
) -> Tuple[str, bool]:
    print('execute_twitter_graph_chat plugin:', plugin.id if plugin else '<none>')
    tz = notification_db.get_user_time_zone(uid)
    result = graph.invoke(
        {
            "uid": uid,
            "tz": tz,
            "cited": cited,
            "messages": messages,
            "plugin_selected": plugin,
            "twitter_context": twitter_context
        },
        {"configurable": {"thread_id": str(uuid.uuid4())}},
    )
    return result.get("answer"), result.get('ask_for_nps', False)


async def execute_twitter_graph_chat_stream(
    uid: str,
    messages: List[Message],
    twitter_context: str,
    plugin: Optional[Plugin] = None,
    cited: Optional[bool] = False,
    callback_data: dict = {}
) -> AsyncGenerator[str, None]:
    print('execute_twitter_graph_chat_stream plugin:', plugin.id if plugin else '<none>')
    tz = notification_db.get_user_time_zone(uid)
    callback = AsyncStreamingCallback()

    task = asyncio.create_task(graph_stream.ainvoke(
        {
            "uid": uid,
            "tz": tz,
            "cited": cited,
            "messages": messages,
            "plugin_selected": plugin,
            "streaming": True,
            "callback": callback,
            "twitter_context": twitter_context
        },
        {"configurable": {"thread_id": str(uuid.uuid4())}},
    ))

    while True:
        try:
            chunk = await callback.queue.get()
            if chunk:
                yield chunk
            else:
                break
        except asyncio.CancelledError:
            break
    await task
    result = task.result()
    callback_data['answer'] = result.get("answer")
    callback_data['ask_for_nps'] = result.get('ask_for_nps', False)

    yield None
    return 