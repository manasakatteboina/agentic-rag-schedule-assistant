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
        collection_name="schedule_initial_collection",
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
    Retrieve schedule information.
    Use this tool for questions about events,
    dates, times, meetings, assignments,
    workshops, appointments and availability.
    """

    # Safety initialization
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
            collection_name="schedule_recovery_collection",
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
# UPDATE / ADD / MOVE / REMOVE TOOL
# ============================================================

@tool
def update_schedule(
    action: str,

    # EXISTING EVENT INFORMATION
    old_title: str = "",
    old_date: str = "",
    old_start_time: str = "",

    # NEW EVENT INFORMATION
    new_title: str = "",
    new_date: str = "",
    new_start_time: str = "",
    new_end_time: str = "",
    new_event_type: str = "",
    new_description: str = ""
) -> str:
    """
    Add, update, move, or remove a schedule event.

    For ADD:
        action = add
        use new_title, new_date, new_start_time, new_end_time

    For UPDATE:
        action = update
        use old_title and old_date to find the event
        and new_* fields for the changes.

    For MOVE:
        action = move
        use old_title, old_date, old_start_time
        and new_date, new_start_time, new_end_time.

    For REMOVE:
        action = remove
        use old_title and old_date.
    """

    # ========================================================
    # MAKE SURE DATA EXISTS
    # ========================================================

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

        if not new_title:
            return "Please provide the event title."

        if not new_date:
            return "Please provide the event date."

        if not new_start_time:
            return "Please provide the event start time."

        new_event = {
            "date": new_date,
            "start_time": new_start_time,
            "end_time": new_end_time,
            "title": new_title,
            "type": (
                new_event_type
                if new_event_type
                else "Task"
            ),
            "description": new_description
        }

        df = pd.concat(
            [
                df,
                pd.DataFrame([new_event])
            ],
            ignore_index=True
        )

        message = (
            f"Added '{new_title}' on {new_date} "
            f"from {new_start_time} to {new_end_time}."
        )


    # ========================================================
    # UPDATE
    # ========================================================

    elif action == "update":

        if not old_title:
            return (
                "Please provide the existing event title "
                "for the update."
            )

        if not old_date:
            return (
                "Please provide the old date of the event."
            )

        # Find existing event
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
                .str.lower()
                ==
                old_start_time.strip().lower()
            )

        if mask.sum() == 0:

            return (
                f"No event found with title "
                f"'{old_title}' on {old_date}."
            )

        # Apply new values
        if new_title:
            df.loc[mask, "title"] = new_title

        if new_date:
            df.loc[mask, "date"] = new_date

        if new_start_time:
            df.loc[mask, "start_time"] = new_start_time

        if new_end_time:
            df.loc[mask, "end_time"] = new_end_time

        if new_event_type:
            df.loc[mask, "type"] = new_event_type

        if new_description:
            df.loc[mask, "description"] = new_description

        message = (
            f"Updated '{old_title}' successfully."
        )


    # ========================================================
    # MOVE
    # ========================================================

    elif action == "move":

        if not old_title:
            return (
                "Please provide the existing event title "
                "for the move."
            )

        if not old_date:
            return (
                "Please provide the old date of the event."
            )

        # Find existing event
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
                .str.lower()
                ==
                old_start_time.strip().lower()
            )

        if mask.sum() == 0:

            return (
                f"No event found with title "
                f"'{old_title}' on {old_date}."
            )

        # Move to new date
        if new_date:
            df.loc[mask, "date"] = new_date

        # Move to new start time
        if new_start_time:
            df.loc[mask, "start_time"] = new_start_time

        # Move to new end time
        if new_end_time:
            df.loc[mask, "end_time"] = new_end_time

        message = (
            f"Moved '{old_title}' from {old_date} "
            f"to {new_date if new_date else old_date}."
        )


    # ========================================================
    # REMOVE
    # ========================================================

    elif action == "remove":

        if not old_title:
            return (
                "Please provide the event title "
                "you want to remove."
            )

        if not old_date:
            return (
                "Please provide the date of the event "
                "you want to remove."
            )

        # Find event
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
                .str.lower()
                ==
                old_start_time.strip().lower()
            )

        if mask.sum() == 0:

            return (
                f"No event found with title "
                f"'{old_title}' on {old_date}."
            )

        df = df[~mask].reset_index(drop=True)

        message = (
            f"Removed '{old_title}' from {old_date}."
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
    # REBUILD VECTOR STORE
    # ========================================================

    new_documents = create_documents(df)

    collection_name = (
        "schedule_collection_"
        + str(abs(hash(str(df.to_dict()))))
    )

    st.session_state.vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings
    )

    st.session_state.vector_store.add_documents(
        new_documents
    )


    return (
        message +
        " Schedule database updated."
    )


# ============================================================
# TOOLS
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

You have two tools:

1. get_schedule
2. update_schedule


============================================================
GET SCHEDULE
============================================================

Use get_schedule when the user asks:

- When is my event?
- When is my assignment?
- What meetings do I have?
- What workshops do I have?
- What appointments do I have?
- What is scheduled?
- Am I free?
- Is there a conflict?


============================================================
UPDATE SCHEDULE
============================================================

Use update_schedule when the user wants to:

- add an event
- update an event
- move an event
- remove an event


============================================================
VERY IMPORTANT
============================================================

The update_schedule tool has TWO sets of information.

OLD EVENT:
old_title
old_date
old_start_time

NEW EVENT:
new_title
new_date
new_start_time
new_end_time
new_event_type
new_description


NEVER confuse old values with new values.


============================================================
MOVE EXAMPLE
============================================================

User:

Move LangChain Workshop from August 27, 2026
2:00 PM to 4:00 PM to August 28, 2026
3:00 PM to 5:00 PM.

Call:

action = "move"

old_title = "LangChain Workshop"

old_date = "2026-08-27"

old_start_time = "02:00 PM"

new_date = "2026-08-28"

new_start_time = "03:00 PM"

new_end_time = "05:00 PM"


============================================================
UPDATE EXAMPLE
============================================================

User:

Update Database Assignment from August 22, 2026
to August 30, 2026.

Call:

action = "update"

old_title = "Database Assignment"

old_date = "2026-08-22"

new_date = "2026-08-30"


============================================================
REMOVE EXAMPLE
============================================================

User:

Remove Database Assignment on August 22, 2026.

Call:

action = "remove"

old_title = "Database Assignment"

old_date = "2026-08-22"


============================================================
ADD EXAMPLE
============================================================

User:

Add Python Assignment on August 30, 2026
from 5:00 PM to 6:00 PM.

Call:

action = "add"

new_title = "Python Assignment"

new_date = "2026-08-30"

new_start_time = "05:00 PM"

new_end_time = "06:00 PM"


============================================================
DATE FORMAT
============================================================

Always convert dates to:

YYYY-MM-DD

Examples:

August 27, 2026
= 2026-08-27

August 28, 2026
= 2026-08-28

August 30, 2026
= 2026-08-30


============================================================
TIME FORMAT
============================================================

Use:

HH:MM AM/PM

Examples:

2 PM
= 02:00 PM

4 PM
= 04:00 PM

5 PM
= 05:00 PM


============================================================
IMPORTANT
============================================================

If the user provides both old and new dates,
do NOT ask for them again.

If the user provides both old and new times,
do NOT ask for them again.

Extract the information from the user's message
and call the correct tool.

Do not invent schedule information.

For schedule questions, use get_schedule.

For changes, use update_schedule.

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
# USER INTERFACE
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


# ============================================================
# PROCESS USER QUERY
# ============================================================

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


    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

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

                answer = (
                    response["messages"][-1].content
                )


                # ------------------------------------------------
                # HANDLE STRUCTURED RESPONSE
                # ------------------------------------------------

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


                # ------------------------------------------------
                # DISPLAY ANSWER
                # ------------------------------------------------

                st.markdown(answer)


            except Exception as e:

                answer = (
                    "Sorry, an error occurred: "
                    + str(e)
                )

                st.error(answer)


    # --------------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
