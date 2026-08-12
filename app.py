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
# GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)


# ============================================================
# CREATE DOCUMENTS (takes a dataframe now, instead of reading a global)
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
# SESSION STATE INITIALIZATION
# This runs only ONCE per browser session (not on every rerun),
# which is the key fix: schedule_df and vector_store now survive
# across every chat message instead of resetting each time.
# ============================================================

if "schedule_df" not in st.session_state:
    st.session_state.schedule_df = pd.DataFrame(
        schedule_data,
        columns=columns
    )

if "vector_store" not in st.session_state:
    documents = create_documents(st.session_state.schedule_df)

    st.session_state.vector_store = Chroma(
        collection_name="schedule_app_collection",
        embedding_function=embeddings
    )
    st.session_state.vector_store.add_documents(documents)


# ============================================================
# GET SCHEDULE TOOL
# ============================================================

@tool
def get_schedule(query: str) -> str:
    """
    Retrieves relevant schedule information based
    on the user's date, time, or question.
    """

    retrieved_docs = st.session_state.vector_store.similarity_search(
        query,
        k=5
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
    Adds, updates, or removes schedule entries.
    """

    df = st.session_state.schedule_df

    # ADD
    if action.lower() == "add":

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

        message = f"Added '{title}' on {date} at {start_time}."

    # UPDATE
    elif action.lower() == "update":

        mask = (
            (df["title"].str.lower() == old_title.lower())
            &
            (df["date"] == old_date)
            &
            (df["start_time"] == old_start_time)
        )

        if mask.sum() == 0:
            return "No matching event found."

        if date:
            df.loc[mask, "date"] = date

        if start_time:
            df.loc[mask, "start_time"] = start_time

        if end_time:
            df.loc[mask, "end_time"] = end_time

        if title:
            df.loc[mask, "title"] = title

        message = "Schedule updated successfully."

    # REMOVE
    elif action.lower() == "remove":

        mask = (
            (df["title"].str.lower() == old_title.lower())
            &
            (df["date"] == old_date)
        )

        if mask.sum() == 0:
            return "No matching event found."

        df = df[~mask]

        message = f"Removed '{old_title}' from the schedule."

    else:
        return "Invalid action. Use add, update, or remove."

    # --------------------------------------------------------
    # SAVE UPDATED DATAFRAME BACK TO SESSION STATE
    # --------------------------------------------------------

    st.session_state.schedule_df = df

    # --------------------------------------------------------
    # REFRESH CHROMADB IN SESSION STATE
    # --------------------------------------------------------

    new_documents = create_documents(df)

    st.session_state.vector_store = Chroma(
        collection_name="schedule_app_collection",
        embedding_function=embeddings
    )

    st.session_state.vector_store.add_documents(new_documents)

    return message + " Schedule database updated."


# ============================================================
# AGENT
# ============================================================

tools = [
    get_schedule,
    update_schedule
]

system_prompt = """
You are an Agentic RAG Schedule Assistant.

You manage the user's schedule for the next 30 days.

You have two tools.

1. get_schedule
Use it when the user asks about:
- existing events
- meetings
- workshops
- tasks
- appointments
- dates
- times
- availability

2. update_schedule
Use it when the user wants to:
- add an event
- update an event
- move an event
- remove an event

Always use get_schedule when you need to
retrieve existing schedule information.

Always use update_schedule when the user
asks to change the schedule.

Do not invent schedule information.

When checking availability, retrieve the
schedule first and identify conflicts.

Answer clearly and simply.
"""

schedule_agent = create_agent(
    llm,
    tools,
    system_prompt=system_prompt
)


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Agentic RAG Schedule Assistant",
    page_icon="📅"
)

st.title("📅 Agentic RAG Schedule Assistant")

st.write(
    "Ask about your schedule or add, update, "
    "move, and remove events."
)


# Chat history

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input

user_query = st.chat_input(
    "Ask about your schedule..."
)


if user_query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    with st.chat_message("user"):
        st.markdown(user_query)


    with st.chat_message("assistant"):

        with st.spinner("Checking your schedule..."):

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

            answer = response["messages"][-1].content

            # Handle Gemini structured content
            if isinstance(answer, list):

                text_parts = []

                for item in answer:

                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(
                                item.get("text", "")
                            )

                    elif isinstance(item, str):
                        text_parts.append(item)

                answer = "\n".join(text_parts)


            st.markdown(answer)


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
