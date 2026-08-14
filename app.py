import os
import re
import uuid
import pandas as pd
import streamlit as st

from langchain_core.documents import Document
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain.agents import create_agent


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Agentic RAG Schedule Assistant",
    page_icon="📅"
)


# ============================================================
# API KEY
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is not configured.")
    st.stop()


# ============================================================
# SAMPLE SCHEDULE
# ============================================================

schedule_data = [
    ["2026-08-12", "10:00 AM", "11:00 AM", "Team Meeting",
     "Meeting", "Discuss project progress"],

    ["2026-08-13", "02:00 PM", "05:00 PM", "AI Workshop",
     "Workshop", "Hands-on Agentic RAG workshop"],

    ["2026-08-14", "09:30 AM", "10:30 AM", "Doctor Appointment",
     "Appointment", "Regular appointment"],

    ["2026-08-15", "02:00 PM", "03:00 PM", "Client Meeting",
     "Meeting", "Discuss project requirements"],

    ["2026-08-16", "11:00 AM", "12:00 PM", "Python Task",
     "Task", "Complete Python practice"],

    ["2026-08-18", "10:00 AM", "12:00 PM", "ML Seminar",
     "Workshop", "Attend machine learning seminar"],

    ["2026-08-20", "01:00 PM", "02:00 PM", "Team Stand-up",
     "Meeting", "Weekly status update"],

    ["2026-08-22", "04:00 PM", "05:00 PM", "Database Assignment",
     "Task", "Complete DBMS assignment"],

    ["2026-08-25", "11:00 AM", "12:30 PM", "Project Review",
     "Meeting", "Review project progress"],

    ["2026-08-27", "02:00 PM", "04:00 PM", "LangChain Workshop",
     "Workshop", "Build a RAG application"],

    ["2026-08-29", "10:00 AM", "11:00 AM", "Career Appointment",
     "Appointment", "Career guidance session"],

    ["2026-09-01", "03:00 PM", "04:30 PM", "Project Demo",
     "Meeting", "Demonstrate final project"],

    ["2026-09-03", "09:00 AM", "10:00 AM", "Documentation Task",
     "Task", "Update project documentation"],

    ["2026-09-05", "01:00 PM", "02:00 PM", "Team Meeting",
     "Meeting", "Sprint planning"],

    ["2026-09-07", "10:00 AM", "12:00 PM", "AI Workshop",
     "Workshop", "Agent evaluation workshop"]
]

columns = [
    "date",
    "start_time",
    "end_time",
    "title",
    "type",
    "description"
]


# ============================================================
# GEMMA MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)


# ============================================================
# CREATE DOCUMENTS
# ============================================================

def create_documents(df):

    docs = []

    for _, row in df.iterrows():

        text = f"""
Date: {row['date']}
Start Time: {row['start_time']}
End Time: {row['end_time']}
Event: {row['title']}
Type: {row['type']}
Description: {row['description']}
"""

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "date": str(row["date"]),
                    "title": str(row["title"]),
                    "type": str(row["type"])
                }
            )
        )

    return docs


# ============================================================
# INITIALIZE SCHEDULE DATA
# ============================================================

if "schedule_df" not in st.session_state:

    st.session_state.schedule_df = pd.DataFrame(
        schedule_data,
        columns=columns
    )


# ============================================================
# INITIALIZE VECTOR STORE
# ============================================================

if "vector_store" not in st.session_state:

    documents = create_documents(
        st.session_state.schedule_df
    )

    st.session_state.vector_store = Chroma(
        collection_name="schedule_app_" + uuid.uuid4().hex,
        embedding_function=embeddings
    )

    if documents:

        st.session_state.vector_store.add_documents(
            documents
        )


# ============================================================
# REFRESH VECTOR STORE
# ============================================================

def refresh_vector_store():

    documents = create_documents(
        st.session_state.schedule_df
    )

    st.session_state.vector_store = Chroma(
        collection_name="schedule_app_" + uuid.uuid4().hex,
        embedding_function=embeddings
    )

    if documents:

        st.session_state.vector_store.add_documents(
            documents
        )


# ============================================================
# GET SCHEDULE TOOL
# ============================================================

@tool
def get_schedule(query: str) -> str:
    """
    Retrieves relevant schedule information.
    """

    if "schedule_df" not in st.session_state:

        st.session_state.schedule_df = pd.DataFrame(
            schedule_data,
            columns=columns
        )

    if "vector_store" not in st.session_state:

        refresh_vector_store()

    retrieved_docs = (
        st.session_state.vector_store
        .similarity_search(query, k=5)
    )

    if not retrieved_docs:

        return "No schedule information found."

    result = ""

    for doc in retrieved_docs:

        result += doc.page_content
        result += "\n" + "-" * 40 + "\n"

    return result


# ============================================================
# UPDATE SCHEDULE TOOL
# ============================================================

@tool
def update_schedule(
    action: str,
    date: str = "",
    start_time: str = "",
    end_time: str = "",
    title: str = "",
    event_type: str = "Task",
    description: str = "",
    old_title: str = "",
    old_date: str = "",
    old_start_time: str = ""
) -> str:
    """
    Adds, updates, moves, or removes schedule events.
    """

    if "schedule_df" not in st.session_state:

        st.session_state.schedule_df = pd.DataFrame(
            schedule_data,
            columns=columns
        )

    df = st.session_state.schedule_df.copy()

    action = action.strip().lower()


    # ========================================================
    # ADD
    # ========================================================

    if action == "add":

        if not date or not start_time or not title:

            return (
                "Please provide date, start time, "
                "and title."
            )

        new_event = {
            "date": date,
            "start_time": start_time,
            "end_time": end_time,
            "title": title,
            "type": event_type,
            "description": description
        }

        df = pd.concat(
            [
                df,
                pd.DataFrame([new_event])
            ],
            ignore_index=True
        )

        message = (
            f"Added '{title}' on {date} "
            f"from {start_time} to {end_time}."
        )


    # ========================================================
    # UPDATE / MOVE
    # ========================================================

    elif action in ["update", "move"]:

        if not old_title or not old_date:

            return (
                "Please provide the old event title "
                "and old date."
            )

        mask = (
            df["title"]
            .astype(str)
            .str.strip()
            .str.lower()
            ==
            old_title.strip().lower()
        ) & (
            df["date"]
            .astype(str)
            .str.strip()
            ==
            old_date.strip()
        )

        if old_start_time:

            mask = mask & (
                df["start_time"]
                .astype(str)
                .str.strip()
                ==
                old_start_time.strip()
            )

        if mask.sum() == 0:

            return (
                f"No matching event found for "
                f"'{old_title}' on {old_date}."
            )

        # Change date
        if date:
            df.loc[mask, "date"] = date

        # Change start time
        if start_time:
            df.loc[mask, "start_time"] = start_time

        # Change end time
        if end_time:
            df.loc[mask, "end_time"] = end_time

        # Change title
        if title:
            df.loc[mask, "title"] = title

        # Change type
        if event_type and event_type != "Task":
            df.loc[mask, "type"] = event_type

        # Change description
        if description:
            df.loc[mask, "description"] = description

        if action == "move":

            message = (
                f"Moved '{old_title}' successfully."
            )

        else:

            message = (
                f"Updated '{old_title}' successfully."
            )


    # ========================================================
    # REMOVE
    # ========================================================

    elif action == "remove":

        if not old_title or not old_date:

            return (
                "Please provide the event title "
                "and date."
            )

        mask = (
            df["title"]
            .astype(str)
            .str.strip()
            .str.lower()
            ==
            old_title.strip().lower()
        ) & (
            df["date"]
            .astype(str)
            .str.strip()
            ==
            old_date.strip()
        )

        if old_start_time:

            mask = mask & (
                df["start_time"]
                .astype(str)
                .str.strip()
                ==
                old_start_time.strip()
            )

        if mask.sum() == 0:

            return (
                f"No matching event found for "
                f"'{old_title}' on {old_date}."
            )

        df = df[~mask].reset_index(drop=True)

        message = (
            f"Removed '{old_title}' "
            f"from {old_date}."
        )


    # ========================================================
    # INVALID ACTION
    # ========================================================

    else:

        return (
            "Invalid action. "
            "Use add, update, move, or remove."
        )


    # ========================================================
    # SAVE UPDATED DATAFRAME
    # ========================================================

    st.session_state.schedule_df = df


    # ========================================================
    # REFRESH RAG DATABASE
    # ========================================================

    refresh_vector_store()


    return (
        message +
        " Schedule database updated."
    )


# ============================================================
# AGENT TOOLS
# ============================================================

tools = [
    get_schedule,
    update_schedule
]


# ============================================================
# SYSTEM PROMPT
# ============================================================

system_prompt = """
You are an Agentic RAG Schedule Assistant.

You manage the user's schedule.

TOOLS:

1. get_schedule

Use this for:
- existing events
- meetings
- workshops
- tasks
- appointments
- dates
- times
- availability
- schedule questions

2. update_schedule

Use this for:
- adding events
- updating events
- moving events
- removing events

IMPORTANT:

Never invent schedule information.

When the user asks about an existing event,
use get_schedule.

When the user wants to change an event,
use update_schedule.

For adding:
action = "add"

For updating:
action = "update"

For moving:
action = "move"

For removing:
action = "remove"

Always actually use the tool when the user
asks to change the schedule.

Do not claim that an event was changed unless
the update_schedule tool was successfully used.

Answer clearly and simply.
"""


# ============================================================
# CREATE AGENT
# ============================================================

schedule_agent = create_agent(
    llm,
    tools,
    system_prompt=system_prompt
)


# ============================================================
# STREAMLIT UI
# ============================================================

st.title(
    "📅 Agentic RAG Schedule Assistant"
)

st.write(
    "Ask about your schedule or add, update, "
    "move, and remove events."
)


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER INPUT
# ============================================================

user_query = st.chat_input(
    "Ask about your schedule..."
)


# ============================================================
# DIRECT COMMAND FUNCTIONS
# ============================================================

def find_event_by_text(date, query):

    df = st.session_state.schedule_df

    matching = df[
        df["date"].astype(str).str.strip() == date
    ]

    query_lower = query.lower()

    for event_title in matching["title"]:

        if str(event_title).lower() in query_lower:

            return str(event_title)

    return None


def extract_dates(text):

    return re.findall(
        r"\d{4}-\d{2}-\d{2}",
        text
    )


def extract_times(text):

    return re.findall(
        r"\b(?:0?[1-9]|1[0-2]):[0-5][0-9]\s*(?:AM|PM)\b",
        text,
        flags=re.IGNORECASE
    )


# ============================================================
# HANDLE USER QUERY
# ============================================================

if user_query:

    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    with st.chat_message("user"):

        st.markdown(user_query)


    query_lower = user_query.lower().strip()

    direct_result = None


    # ========================================================
    # REMOVE
    # ========================================================

    if query_lower.startswith("remove"):

        dates = extract_dates(user_query)

        if dates:

            old_date = dates[0]

            event_title = find_event_by_text(
                old_date,
                user_query
            )

            if event_title:

                direct_result = update_schedule.invoke(
                    {
                        "action": "remove",
                        "old_title": event_title,
                        "old_date": old_date
                    }
                )

            else:

                direct_result = (
                    "I could not identify the event "
                    f"on {old_date}. Please include "
                    "the event title."
                )

        else:

            direct_result = (
                "Please provide the date of the "
                "event you want to remove."
            )


    # ========================================================
    # MOVE
    # ========================================================

    elif query_lower.startswith("move"):

        dates = extract_dates(user_query)

        if len(dates) >= 2:

            old_date = dates[0]
            new_date = dates[1]

            event_title = find_event_by_text(
                old_date,
                user_query
            )

            if event_title:

                times = extract_times(user_query)

                arguments = {
                    "action": "move",
                    "old_title": event_title,
                    "old_date": old_date,
                    "date": new_date
                }

                # If two times are included,
                # treat them as new start/end times.
                if len(times) >= 2:

                    arguments["start_time"] = times[0]
                    arguments["end_time"] = times[1]

                elif len(times) == 1:

                    arguments["start_time"] = times[0]

                direct_result = update_schedule.invoke(
                    arguments
                )

            else:

                direct_result = (
                    f"I could not find an event on "
                    f"{old_date}. Please include "
                    "the exact event title."
                )

        else:

            direct_result = (
                "For moving an event, provide the "
                "old date and new date."
            )


    # ========================================================
    # UPDATE
    # ========================================================

    elif query_lower.startswith("update"):

        dates = extract_dates(user_query)

        if len(dates) >= 2:

            old_date = dates[0]
            new_date = dates[1]

            event_title = find_event_by_text(
                old_date,
                user_query
            )

            if event_title:

                times = extract_times(user_query)

                arguments = {
                    "action": "update",
                    "old_title": event_title,
                    "old_date": old_date,
                    "date": new_date
                }

                if len(times) >= 2:

                    arguments["start_time"] = times[0]
                    arguments["end_time"] = times[1]

                elif len(times) == 1:

                    arguments["start_time"] = times[0]

                direct_result = update_schedule.invoke(
                    arguments
                )

            else:

                direct_result = (
                    f"I could not find an event on "
                    f"{old_date}. Please include "
                    "the exact event title."
                )

        else:

            direct_result = (
                "For updating an event, provide the "
                "old date and new date."
            )


    # ========================================================
    # SHOW DIRECT RESULT
    # ========================================================

    if direct_result is not None:

        with st.chat_message("assistant"):

            st.markdown(direct_result)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": direct_result
            }
        )


    # ========================================================
    # NORMAL AGENT / RAG QUESTIONS
    # ========================================================

    else:

        with st.chat_message("assistant"):

            with st.spinner(
                "Checking your schedule..."
            ):

                try:

                    response = schedule_agent.invoke(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": user_query
                                }
                            ]
                        }
                    )

                    answer = response[
                        "messages"
                    ][-1].content


                    # --------------------------------------------
                    # HANDLE STRUCTURED CONTENT
                    # --------------------------------------------

                    if isinstance(answer, list):

                        text_parts = []

                        for item in answer:

                            if isinstance(
                                item,
                                dict
                            ):

                                if item.get(
                                    "type"
                                ) == "text":

                                    text_parts.append(
                                        item.get(
                                            "text",
                                            ""
                                        )
                                    )

                            elif isinstance(
                                item,
                                str
                            ):

                                text_parts.append(
                                    item
                                )

                        answer = "\n".join(
                            text_parts
                        )


                    st.markdown(answer)


                except Exception as e:

                    answer = (
                        "Sorry, an error occurred: "
                        + str(e)
                    )

                    st.error(answer)


        # ----------------------------------------------------
        # SAVE ASSISTANT MESSAGE
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )
