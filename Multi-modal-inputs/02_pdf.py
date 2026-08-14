from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from rich import print

from dotenv import load_dotenv

import os
import base64
import httpx


# ============================================================
# Environment
# ============================================================

load_dotenv()


model = ChatOpenAI(
    model="MiniMax-M3",
    base_url=os.getenv("MINIMAX_BASE_URL"),
    api_key=os.getenv("MINIMAX_API_KEY"),
    temperature=0.2,
)



# ============================================================
# Deep Agent
# ============================================================

agent = create_deep_agent(
    model=model
)



# ============================================================
# Helpers
# ============================================================

def fetch_file_as_base64(url: str):

    response = httpx.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    return base64.b64encode(
        response.content
    ).decode("utf-8")



def encode_local_file(path):

    with open(
        path,
        "rb"
    ) as f:

        return base64.b64encode(
            f.read()
        ).decode("utf-8")



# ============================================================
# 1. Remote PDF
# ============================================================


print(
    "[bold green]Remote PDF test[/bold green]"
)


pdf_url = (
    "https://ik.imagekit.io/miyasajid19/resume.pdf"
)


pdf_base64 = fetch_file_as_base64(
    pdf_url
)



result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": [

                    {
                        "type": "text",
                        "text":
                        "Summarize this PDF document."
                    },


                    {
                        "type": "file",
                        "filename":
                        "resume.pdf",

                        "mime_type":
                        "application/pdf",

                        "base64":
                        pdf_base64,
                    }

                ],
            }
        ]
    }
)



print(
    result["messages"][-1].content
)




# ============================================================
# 2. Local Files
# ============================================================


print(
    "[bold green]Local files test[/bold green]"
)



resume_path = (
    r"C:\Users\miyas\Downloads\Sajid_Miya_Resume_2.pdf"
)


docx_path = (
    r"C:\Users\miyas\Downloads\UTA025.docx"
)



resume_base64 = encode_local_file(
    resume_path
)


docx_base64 = encode_local_file(
    docx_path
)



result = agent.invoke(
    {
        "messages": [

            {
                "role":
                "user",

                "content":

                [

                    {
                        "type":
                        "text",

                        "text":
                        """
Analyze these two documents.

1. Explain what each document contains.
2. Compare their purpose.
3. Extract important information.
"""
                    },


                    {
                        "type":
                        "file",

                        "filename":
                        "resume.pdf",

                        "mime_type":
                        "application/pdf",

                        "base64":
                        resume_base64,
                    },


                    {
                        "type":
                        "file",

                        "filename":
                        "document.docx",

                        "mime_type":
                        (
                            "application/vnd."
                            "openxmlformats-officedocument."
                            "wordprocessingml.document"
                        ),

                        "base64":
                        docx_base64,
                    }

                ]
            }

        ]
    }
)



print(
    result["messages"][-1].content
)




# ============================================================
# 3. Hybrid Remote + Local
# ============================================================


print(
    "[bold green]Hybrid test[/bold green]"
)



result = agent.invoke(
    {
        "messages":

        [

            {
                "role":
                "user",

                "content":

                [

                    {
                        "type":
                        "text",

                        "text":
                        """
Compare these two resumes.
Identify differences.
"""
                    },


                    {
                        "type":
                        "file",

                        "filename":
                        "online_resume.pdf",

                        "mime_type":
                        "application/pdf",

                        "base64":
                        fetch_file_as_base64(
                            pdf_url
                        ),
                    },


                    {
                        "type":
                        "file",

                        "filename":
                        "local_resume.pdf",

                        "mime_type":
                        "application/pdf",

                        "base64":
                        resume_base64,
                    }

                ]
            }

        ]
    }
)



print(
    result["messages"][-1].content
)