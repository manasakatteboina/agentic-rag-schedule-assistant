import os
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
# SAMPLE 30-DAY SCHEDULE
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
# GEMMA 4 LLM
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
                    "date": row["date"],
                    "title": row["title"],
                    "type": row["type"]
                }
            )
        )

    return docs


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

# IMPORTANT:
# schedule_df is created BEFORE any tool can use it.

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
        collection_name="schedule_app_collection",
        embedding_function=embeddings
    )

    st.session_state.vector_store.add_documents(
        documents
    )


# ============================================================
# GET SCHEDULE TOOL
# ============================================================

@tool
def get_schedule(query: str) -> str:
    """
    Retrieves relevant schedule information based
    on the user's date, time, or question.
    """

    # Safety check
    if "schedule_df" not in st.session_state:

        st.session_state.schedule_df = pd.DataFrame(
            schedule_data,
            columns=columns
        )

    if "vector_store" not in st.session_state:

        documents = create_documents(
            st.session_state.schedule_df
        )

        st.session_state.vector_store = Chroma(
            collection_name="schedule_app_collection",
            embedding_function=embeddings
        )

        st.session_state.vector_store.add_documents(
            documents
        )

    retrieved_docs = (
        st.session_state.vector_store.similarity_search(
            query,
            k=5
        )
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
    Adds, updates, moves, or removes schedule entries.
    """

    # Safety check
    if "schedule_df" not in st.session_state:

        st.session_state.schedule_df = pd.DataFrame(
            schedule_data,
            columns=columns
        )

    df = st.session_state.schedule_df.copy()

    action = action.lower().strip()


    # ========================================================
    # ADD
    # ========================================================

    if action == "add":

        if not date or not start_time or not title:

            return (
                "To add an event, please provide "
                "date, start time, and title."
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
            f"at {start_time}."
        )


    # ========================================================
    # UPDATE / MOVE
    # ========================================================

    elif action in ["update", "move"]:

        if not old_title or not old_date:

            return (
                "To update or move an event, "
                "please provide the old event title "
                "and old date."
            )

        mask = (
            (df["title"].astype(str).str.lower()
             == old_title.lower())
            &
            (df["date"].astype(str)
             == old_date)
        )

        # If old start time was supplied,
        # use it to identify the exact event.

        if old_start_time:

            mask = (
                mask
                &
                (
                    df["start_time"].astype(str)
                    == old_start_time
                )
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

        if event_type:

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
                "To remove an event, please provide "
                "the event title and date."
            )

        mask = (
            (df["title"].astype(str).str.lower()
             == old_title.lower())
            &
            (df["date"].astype(str)
             == old_date)
        )

        if old_start_time:

            mask = (
                mask
                &
                (
                    df["start_time"].astype(str)
                    == old_start_time
                )
            )

        if mask.sum() == 0:

            return (
                f"No matching event found for "
                f"'{old_title}' on {old_date}."
            )

        df = df[~mask].reset_index(drop=True)

        message = (
            f"Removed '{old_title}' from "
            f"{old_date}."
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
    # SAVE DATAFRAME
    # ========================================================

    st.session_state.schedule_df = df


    # ========================================================
    # REBUILD CHROMA VECTOR STORE
    # ========================================================

    new_documents = create_documents(df)

    # Use a new collection name every time we refresh.
    # This avoids old records remaining in Chroma.

    collection_name = (
        "schedule_app_collection_"
        + str(len(df))
    )

    st.session_state.vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings
    )

    st.session_state.vector_store.add_documents(
        new_documents
    )


    return message + " Schedule database updated."


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

You have two tools.

1. get_schedule

Use get_schedule when the user asks about:

- existing events
- meetings
- workshops
- tasks
- appointments
- dates
- times
- availability
- schedule information

2. update_schedule

Use update_schedule when the user wants to:

- add an event
- update an event
- move an event
- remove an event

IMPORTANT RULES:

Always use get_schedule when you need to
retrieve existing schedule information.

Always use update_schedule when the user
asks to change the schedule.

Never invent schedule information.

When checking availability, first retrieve
the schedule and identify conflicts.

For moving an event, use action="move".

For changing event information, use
action="update".

For adding an event, use action="add".

For deleting an event, use action="remove".

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

st.title("📅 Agentic RAG Schedule Assistant")

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

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# USER INPUT
# ============================================================

user_query = st.chat_input(
    "Ask about your schedule..."
)


if user_query:

    # Save user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    with st.chat_message("user"):

        st.markdown(user_query)


    # ========================================================
    # ASSISTANT RESPONSE
    # ========================================================

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


                # ====================================================
                # HANDLE GEMINI / GEMMA STRUCTURED CONTENT
                # ====================================================

                if isinstance(answer, list):

                    text_parts = []

                    for item in answer:

                        if isinstance(item, dict):

                            if item.get("type") == "text":

                                text_parts.append(
                                    item.get(
                                        "text",
                                        ""
                                    )
                                )

                        elif isinstance(item, str):

                            text_parts.append(item)

                    answer = "\n".join(
                        text_parts
                    )


                # ====================================================
                # FINAL ANSWER
                # ====================================================

                st.markdown(answer)


            except Exception as e:

                answer = (
                    "Sorry, an error occurred: "
                    + str(e)
                )

                st.error(answer)


    # Save assistant response

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
