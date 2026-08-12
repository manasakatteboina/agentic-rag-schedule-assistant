import os
import re
from datetime import datetime, date, timedelta

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
# STREAMLIT PAGE
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
    [
        "2026-08-12",
        "10:00 AM",
        "11:00 AM",
        "Team Meeting",
        "Meeting",
        "Discuss project progress"
    ],
    [
        "2026-08-13",
        "02:00 PM",
        "05:00 PM",
        "AI Workshop",
        "Workshop",
        "Hands-on Agentic RAG workshop"
    ],
    [
        "2026-08-14",
        "09:30 AM",
        "10:30 AM",
        "Doctor Appointment",
        "Appointment",
        "Regular appointment"
    ],
    [
        "2026-08-15",
        "02:00 PM",
        "03:00 PM",
        "Client Meeting",
        "Meeting",
        "Discuss project requirements"
    ],
    [
        "2026-08-16",
        "11:00 AM",
        "12:00 PM",
        "Python Task",
        "Task",
        "Complete Python practice"
    ],
    [
        "2026-08-18",
        "10:00 AM",
        "12:00 PM",
        "ML Seminar",
        "Workshop",
        "Attend machine learning seminar"
    ],
    [
        "2026-08-20",
        "01:00 PM",
        "02:00 PM",
        "Team Stand-up",
        "Meeting",
        "Weekly status update"
    ],
    [
        "2026-08-22",
        "04:00 PM",
        "05:00 PM",
        "Database Assignment",
        "Task",
        "Complete DBMS assignment"
    ],
    [
        "2026-08-25",
        "11:00 AM",
        "12:30 PM",
        "Project Review",
        "Meeting",
        "Review project progress"
    ],
    [
        "2026-08-27",
        "02:00 PM",
        "04:00 PM",
        "LangChain Workshop",
        "Workshop",
        "Build a RAG application"
    ],
    [
        "2026-08-29",
        "10:00 AM",
        "11:00 AM",
        "Career Appointment",
        "Appointment",
        "Career guidance session"
    ],
    [
        "2026-09-01",
        "03:00 PM",
        "04:30 PM",
        "Project Demo",
        "Meeting",
        "Demonstrate final project"
    ],
    [
        "2026-09-03",
        "09:00 AM",
        "10:00 AM",
        "Documentation Task",
        "Task",
        "Update project documentation"
    ],
    [
        "2026-09-05",
        "01:00 PM",
        "02:00 PM",
        "Team Meeting",
        "Meeting",
        "Sprint planning"
    ],
    [
        "2026-09-07",
        "10:00 AM",
        "12:00 PM",
        "AI Workshop",
        "Workshop",
        "Agent evaluation workshop"
    ]
]


COLUMNS = [
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
# CREATE DOCUMENTS
# ============================================================

def create_documents(df):

    documents = []

    for _, row in df.iterrows():

        text = f"""
Date: {row['date']}
Start Time: {row['start_time']}
End Time: {row['end_time']}
Event: {row['title']}
Type: {row['type']}
Description: {row['description']}
"""

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "date": str(row["date"]),
                    "title": str(row["title"]),
                    "type": str(row["type"])
                }
            )
        )

    return documents


# ============================================================
# SCHEDULE STATE
#
# IMPORTANT:
# We do NOT use st.session_state inside the LangChain tools.
# This fixes the Render/LangGraph error.
# ============================================================

class ScheduleState:

    def __init__(self):

        self.df = pd.DataFrame(
            schedule_data,
            columns=COLUMNS
        )

        self.vector_store = None

        self.create_vector_store()

    def create_vector_store(self):

        # Create a unique collection name.
        # This avoids conflicts between deployments/sessions.
        collection_name = (
            "schedule_app_"
            + datetime.now().strftime("%Y%m%d%H%M%S%f")
        )

        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings
        )

        documents = create_documents(self.df)

        if documents:

            self.vector_store.add_documents(
                documents
            )

    def rebuild_vector_store(self):

        self.create_vector_store()


# ============================================================
# GLOBAL SCHEDULE STATE
# ============================================================

if "schedule_state" not in st.session_state:

    st.session_state.schedule_state = ScheduleState()


schedule_state = st.session_state.schedule_state


# ============================================================
# HELPER: FORMAT EVENTS
# ============================================================

def format_events(df):

    if df.empty:
        return "No schedule information found."

    result = []

    for _, row in df.iterrows():

        result.append(
            f"Date: {row['date']}\n"
            f"Time: {row['start_time']} - {row['end_time']}\n"
            f"Event: {row['title']}\n"
            f"Type: {row['type']}\n"
            f"Description: {row['description']}"
        )

    return "\n\n" + ("\n" + "-" * 45 + "\n").join(result)


# ============================================================
# HELPER: FIND DATE FROM QUERY
# ============================================================

def find_date_in_query(query):

    q = query.lower()

    today = date.today()

    # Today
    if "today" in q:
        return today

    # Tomorrow
    if "tomorrow" in q:
        return today + timedelta(days=1)

    # Day after tomorrow
    if "day after tomorrow" in q:
        return today + timedelta(days=2)

    # Weekdays
    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6
    }

    for name, weekday_number in weekdays.items():

        if name in q:

            days_ahead = (
                weekday_number - today.weekday()
            ) % 7

            # If today is that weekday,
            # use the next occurrence.
            if days_ahead == 0:
                days_ahead = 7

            return today + timedelta(
                days=days_ahead
            )

    # YYYY-MM-DD
    match = re.search(
        r"\b(2026-\d{2}-\d{2})\b",
        q
    )

    if match:

        try:
            return datetime.strptime(
                match.group(1),
                "%Y-%m-%d"
            ).date()

        except ValueError:
            pass

    # August 15 / August 15th
    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12
    }

    for month_name, month_number in months.items():

        pattern = (
            rf"\b{month_name}\s+"
            rf"(\d{{1,2}})(?:st|nd|rd|th)?\b"
        )

        match = re.search(
            pattern,
            q
        )

        if match:

            day_number = int(
                match.group(1)
            )

            try:

                result = date(
                    today.year,
                    month_number,
                    day_number
                )

                return result

            except ValueError:
                pass

    return None


# ============================================================
# HELPER: TIME PERIOD
# ============================================================

def get_time_period(query):

    q = query.lower()

    if "morning" in q:
        return 6, 12

    if "afternoon" in q:
        return 12, 17

    if "evening" in q:
        return 17, 21

    if "night" in q:
        return 21, 24

    return None


# ============================================================
# HELPER: CONVERT TIME TO MINUTES
# ============================================================

def time_to_minutes(time_string):

    try:

        parsed = datetime.strptime(
            time_string.strip(),
            "%I:%M %p"
        )

        return (
            parsed.hour * 60
            + parsed.minute
        )

    except Exception:

        return None


# ============================================================
# TOOL 1: GET SCHEDULE
# ============================================================

@tool
def get_schedule(query: str) -> str:
    """
    Retrieves relevant schedule information based
    on date, time, availability, or user question.
    """

    df = schedule_state.df.copy()

    # --------------------------------------------------------
    # Date-based retrieval
    # --------------------------------------------------------

    target_date = find_date_in_query(query)

    if target_date:

        date_string = target_date.strftime(
            "%Y-%m-%d"
        )

        date_df = df[
            df["date"] == date_string
        ]

    else:

        date_df = df


    # --------------------------------------------------------
    # Time-period filtering
    # --------------------------------------------------------

    time_period = get_time_period(query)

    if time_period and not date_df.empty:

        period_start, period_end = time_period

        matching_rows = []

        for index, row in date_df.iterrows():

            start_minutes = time_to_minutes(
                row["start_time"]
            )

            end_minutes = time_to_minutes(
                row["end_time"]
            )

            if (
                start_minutes is not None
                and end_minutes is not None
                and start_minutes < period_end * 60
                and end_minutes > period_start * 60
            ):

                matching_rows.append(index)

        date_df = date_df.loc[
            matching_rows
        ]


    # --------------------------------------------------------
    # Exact date/time result
    # --------------------------------------------------------

    if not date_df.empty:

        return format_events(
            date_df
        )


    # --------------------------------------------------------
    # RAG RETRIEVAL
    # --------------------------------------------------------

    try:

        retrieved_docs = (
            schedule_state.vector_store
            .similarity_search(
                query,
                k=5
            )
        )

        if retrieved_docs:

            result = []

            for doc in retrieved_docs:

                result.append(
                    doc.page_content
                )

            return (
                "\n\n"
                + ("\n" + "-" * 45 + "\n")
                .join(result)
            )

    except Exception as e:

        return (
            "Unable to retrieve schedule "
            f"information: {str(e)}"
        )

    return "No schedule information found."


# ============================================================
# TOOL 2: UPDATE SCHEDULE
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
    Adds, updates/moves, or removes schedule entries.

    action must be:
    add
    update
    remove
    """

    global schedule_state

    df = schedule_state.df.copy()

    action = action.lower().strip()


    # ========================================================
    # ADD
    # ========================================================

    if action == "add":

        if not date:
            return "Please provide the date."

        if not start_time:
            return "Please provide the start time."

        if not title:
            return "Please provide the event title."

        if not end_time:

            end_time = start_time

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
                pd.DataFrame(
                    [new_event]
                )
            ],
            ignore_index=True
        )

        message = (
            f"Added '{title}' on "
            f"{date} at {start_time}."
        )


    # ========================================================
    # UPDATE / MOVE
    # ========================================================

    elif action == "update":

        if not old_title:
            return (
                "Please provide the existing "
                "event title."
            )

        # ----------------------------------------------------
        # Find matching event
        # ----------------------------------------------------

        mask = (
            df["title"]
            .str.lower()
            .str.strip()
            ==
            old_title.lower().strip()
        )

        # Filter by old date if provided
        if old_date:

            mask = (
                mask
                &
                (df["date"] == old_date)
            )

        # Filter by old start time if provided
        if old_start_time:

            mask = (
                mask
                &
                (
                    df["start_time"]
                    == old_start_time
                )
            )

        if mask.sum() == 0:

            return (
                "No matching event found. "
                "Please check the event name, "
                "date, or time."
            )

        # ----------------------------------------------------
        # Update fields
        # ----------------------------------------------------

        if date:
            df.loc[mask, "date"] = date

        if start_time:
            df.loc[
                mask,
                "start_time"
            ] = start_time

        if end_time:
            df.loc[
                mask,
                "end_time"
            ] = end_time

        if title:
            df.loc[
                mask,
                "title"
            ] = title

        if event_type:
            df.loc[
                mask,
                "type"
            ] = event_type

        if description:
            df.loc[
                mask,
                "description"
            ] = description

        message = (
            f"Updated '{old_title}' successfully."
        )


    # ========================================================
    # REMOVE
    # ========================================================

    elif action == "remove":

        if not old_title:

            return (
                "Please provide the event "
                "title to remove."
            )

        mask = (
            df["title"]
            .str.lower()
            .str.strip()
            ==
            old_title.lower().strip()
        )

        if old_date:

            mask = (
                mask
                &
                (df["date"] == old_date)
            )

        if old_start_time:

            mask = (
                mask
                &
                (
                    df["start_time"]
                    == old_start_time
                )
            )

        if mask.sum() == 0:

            return (
                "No matching event found."
            )

        df = df[
            ~mask
        ].reset_index(
            drop=True
        )

        message = (
            f"Removed '{old_title}' "
            "from the schedule."
        )


    # ========================================================
    # INVALID ACTION
    # ========================================================

    else:

        return (
            "Invalid action. "
            "Use add, update, or remove."
        )


    # ========================================================
    # SAVE DATA
    # ========================================================

    schedule_state.df = df


    # ========================================================
    # REBUILD CHROMA DATABASE
    # ========================================================

    try:

        schedule_state.rebuild_vector_store()

    except Exception as e:

        return (
            message
            + " However, the schedule was changed "
            + f"but the RAG database failed to refresh: {str(e)}"
        )


    return (
        message
        + " Schedule database updated."
    )


# ============================================================
# AGENT TOOLS
# ============================================================

tools = [
    get_schedule,
    update_schedule
]


# ============================================================
# CURRENT DATE
# ============================================================

TODAY = date.today().strftime(
    "%Y-%m-%d"
)


# ============================================================
# AGENT SYSTEM PROMPT
# ============================================================

system_prompt = f"""
You are an Agentic RAG Schedule Assistant.

Today's date is {TODAY}.

You manage the user's schedule for the next 30 days.

You have exactly two tools.

============================================================
TOOL 1: get_schedule
============================================================

Use get_schedule when the user asks about:

- existing events
- meetings
- workshops
- tasks
- appointments
- dates
- times
- availability
- free time
- what is scheduled
- schedule conflicts

Always retrieve the schedule instead of guessing.

============================================================
TOOL 2: update_schedule
============================================================

Use update_schedule when the user wants to change
the schedule.

Supported actions:

1. add
2. update
3. remove

For ADD:

Use:
action="add"

For UPDATE or MOVE:

Use:
action="update"

For REMOVE:

Use:
action="remove"

============================================================
IMPORTANT RULES
============================================================

Never invent schedule information.

When the user asks whether they are free,
use get_schedule first.

When the user asks to move an event,
find the existing event and then use update_schedule.

When the user says:

"Move my meeting from 2 PM to 4 PM"

you should identify the existing meeting and update
its start time to 4 PM.

When the user says:

"Add a meeting on August 15 at 3 PM"

add a new schedule entry.

When the user says:

"Remove my Client Meeting"

remove the matching event.

Give short, clear answers.

Do not expose tool arguments to the user.

============================================================
EXAMPLES
============================================================

User:
What do I have scheduled tomorrow?

Use:
get_schedule

User:
Am I free Friday afternoon?

Use:
get_schedule

User:
Add a meeting on August 15 at 3 PM.

Use:
update_schedule with action="add"

User:
Move my Client Meeting from 2 PM to 4 PM.

Use:
update_schedule with action="update"

User:
Remove my Client Meeting on August 15.

Use:
update_schedule with action="remove"
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
# CHAT INPUT
# ============================================================

user_query = st.chat_input(
    "Ask about your schedule..."
)


if user_query:

    # --------------------------------------------------------
    # SHOW USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    with st.chat_message("user"):

        st.markdown(
            user_query
        )


    # --------------------------------------------------------
    # AGENT RESPONSE
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
                    response["messages"][-1]
                    .content
                )


                # --------------------------------------------
                # HANDLE GEMINI STRUCTURED CONTENT
                # --------------------------------------------

                if isinstance(
                    answer,
                    list
                ):

                    text_parts = []

                    for item in answer:

                        if isinstance(
                            item,
                            dict
                        ):

                            if (
                                item.get("type")
                                == "text"
                            ):

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


                if not answer:

                    answer = (
                        "I could not generate "
                        "a response."
                    )


                # --------------------------------------------
                # DISPLAY
                # --------------------------------------------

                st.markdown(
                    answer
                )


            except Exception as e:

                answer = (
                    "Sorry, an error occurred: "
                    + str(e)
                )

                st.error(
                    answer
                )


    # --------------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
