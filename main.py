"""
Multi-Agent Blog Generator using LangChain (RunnableSequence)
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_core.runnables import RunnableSequence, RunnableLambda

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"🔑 OpenAI API Key Loaded: {'Yes' if OPENAI_API_KEY else 'No'}")

app = FastAPI(title="Multi-Agent Blog Generator (LangChain RunnableSequence)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class BlogRequest(BaseModel):
    topic: str
    tone: str = "professional"
    length: str = "medium"  # short, medium, long


class BlogResponse(BaseModel):
    topic: str
    research: str
    draft: str
    final_blog: str
    status: str

# -----------------------------------------------------------------------------
# LLM
# -----------------------------------------------------------------------------


def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)


llm = get_llm()

# -----------------------------------------------------------------------------
# Agents as functions
# -----------------------------------------------------------------------------


def research_agent(inputs):
    topic = inputs["topic"]
    tone = inputs["tone"]
    length = inputs["length"]

    prompt = f"""
You are a research agent. Research the topic: "{topic}"

Provide:
1. Key points to cover
2. Important facts and statistics
3. Structured outline for a {length} blog post
4. Relevant angles and perspectives

Tone: {tone}
    """

    result = llm.invoke([HumanMessage(content=prompt)])
    return {**inputs, "research": result.content}


def writer_agent(inputs):
    word_count = {
        "short": "500-700",
        "medium": "800-1200",
        "long": "1500-2000"
    }

    prompt = f"""
You are a professional blog writer. Write a complete blog post based on the following research:

Research:
{inputs['research']}

Requirements:
- Topic: {inputs['topic']}
- Tone: {inputs['tone']}
- Length: {word_count[inputs['length']]} words
- Engaging introduction, clear headings, strong conclusion, SEO-friendly.
    """

    result = llm.invoke([HumanMessage(content=prompt)])
    return {**inputs, "draft": result.content}


def reviewer_agent(inputs):
    prompt = f"""
You are an editorial reviewer. Review and polish this draft:

Draft:
{inputs['draft']}

Check for:
1. Grammar and spelling
2. Flow and readability
3. Tone consistency ({inputs['tone']})
4. SEO optimization

Return the final polished version.
    """

    result = llm.invoke([HumanMessage(content=prompt)])
    return {**inputs, "final_blog": result.content}


# -----------------------------------------------------------------------------
# Create LangChain RunnableSequence
# -----------------------------------------------------------------------------

blog_chain = RunnableLambda(research_agent) | RunnableLambda(
    writer_agent) | RunnableLambda(reviewer_agent)


# -----------------------------------------------------------------------------
# API Routes
# -----------------------------------------------------------------------------


@app.get("/")
def root():
    return {
        "message": "Multi-Agent Blog Generator API (LangChain Chaining)",
        "endpoints": {
            "/generate": "POST - Generate blog",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "3.0.0"}


@app.post("/generate", response_model=BlogResponse)
async def generate_blog(request: BlogRequest):
    try:
        inputs = {
            "topic": request.topic,
            "tone": request.tone,
            "length": request.length
        }

        final_state = blog_chain.invoke(inputs)

        return BlogResponse(
            topic=request.topic,
            research=final_state.get("research", ""),
            draft=final_state.get("draft", ""),
            final_blog=final_state.get("final_blog", ""),
            status="success"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# Run Server
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))