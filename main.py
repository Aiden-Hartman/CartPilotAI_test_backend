from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
import traceback
from typing import Dict, Any
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiting settings
REQUESTS_PER_MIN = 3  # Free tier limit
request_timestamps = []

app = FastAPI()

# Allow frontend to access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can lock this down to your frontend domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "✅ CartPilotAI backend is running!",
        "docs": "/docs",
        "post_endpoint": "/chat"
    }

def check_rate_limit():
    """Implements a rolling window rate limit"""
    global request_timestamps
    
    now = time.time()
    # Remove timestamps older than 1 minute
    request_timestamps = [ts for ts in request_timestamps if now - ts < 60]
    
    if len(request_timestamps) >= REQUESTS_PER_MIN:
        time_to_wait = 60 - (now - request_timestamps[0])
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Please wait {int(time_to_wait)} seconds."
        )
    
    request_timestamps.append(now)

try:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is not set!")
        raise ValueError("OPENAI_API_KEY not found")
    
    client = OpenAI(api_key=api_key)
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    logger.error(traceback.format_exc())
    raise

class ChatRequest(BaseModel):
    message: str
    client_id: str  # for future use

def handle_openai_error(e: Exception) -> HTTPException:
    """Centralized error handling for OpenAI API errors"""
    error_msg = str(e)
    status_code = 500
    
    if "exceeded your current quota" in error_msg:
        logger.error("API quota exceeded. Please check billing settings.")
        return HTTPException(
            status_code=429,
            detail="API quota exceeded. Please ensure billing is set up correctly."
        )
    elif "rate limit" in error_msg.lower():
        logger.error("Rate limit hit")
        return HTTPException(
            status_code=429,
            detail="Too many requests. Please try again in a few seconds."
        )
    elif "invalid_api_key" in error_msg:
        logger.error("Invalid API key")
        return HTTPException(
            status_code=401,
            detail="Invalid API key. Please check your configuration."
        )
    elif "context_length_exceeded" in error_msg:
        logger.error("Context length exceeded")
        return HTTPException(
            status_code=400,
            detail="Input too long. Please send a shorter message."
        )
    
    logger.error(f"Unexpected OpenAI error: {error_msg}")
    return HTTPException(status_code=status_code, detail=f"OpenAI API error: {error_msg}")

def log_openai_response(response: Dict[Any, Any], step: str) -> None:
    """Helper function to log OpenAI response details"""
    logger.info(f"OpenAI Response ({step}):")
    logger.info(f"Finish Reason: {response.choices[0].finish_reason}")
    logger.info(f"Model Used: {response.model}")
    logger.info(f"Usage: {response.usage}")
    # Log token usage for monitoring
    if hasattr(response, 'usage'):
        logger.info(f"Tokens used - Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}")

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        # Check rate limit first
        check_rate_limit()
        
        logger.info(f"Received chat request from client {req.client_id}")
        logger.info(f"User message: {req.message}")
        
        # Step 1: Initial GPT call
        logger.info("Making initial GPT call...")
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using 3.5 for free tier compatibility
                messages=[
                    {"role": "system", "content": "You are a helpful shopping assistant."},
                    {"role": "user", "content": req.message}
                ],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "recommend_products",
                        "description": "Recommend products based on query",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"}
                            },
                            "required": ["query"]
                        }
                    }
                }],
                timeout=30  # Add timeout to prevent hanging
            )
            log_openai_response(response, "initial")
            
        except Exception as e:
            raise handle_openai_error(e)

        # Step 2: Tool call?
        if response.choices[0].finish_reason == "tool_calls":
            logger.info("Tool call detected, processing...")
            try:
                # Extract tool call details
                tool_call = response.choices[0].message.tool_calls[0]
                tool_call_id = tool_call.id
                tool_args = json.loads(tool_call.function.arguments)
                tool_query = tool_args['query']
                
                logger.info(f"Tool call ID: {tool_call_id}")
                logger.info(f"Tool call details: {tool_call.function.name}")
                logger.info(f"Tool query: {tool_query}")

                # Step 3: Run the tool (replace with real DB query later)
                logger.info("Generating fake results (to be replaced with real DB query)")
                fake_results = {
                    "products": [
                        {"title": "Ashwagandha", "price": "$21.99", "summary": "A calming herb."}
                    ]
                }

                # Step 4: Call GPT again with tool result
                logger.info("Making final GPT call with tool results...")
                try:
                    final = client.chat.completions.create(
                        model="gpt-3.5-turbo",  # Using 3.5 for free tier compatibility
                        messages=[
                            {"role": "system", "content": "You are a helpful shopping assistant."},
                            {"role": "user", "content": req.message},
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": tool_call_id,
                                        "type": "function",
                                        "function": {
                                            "name": "recommend_products",
                                            "arguments": json.dumps({"query": tool_query})
                                        }
                                    }
                                ]
                            },
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": json.dumps(fake_results)
                            }
                        ],
                        timeout=30  # Add timeout to prevent hanging
                    )
                    log_openai_response(final, "final")
                    
                except Exception as e:
                    raise handle_openai_error(e)

                logger.info("Successfully generated final response")
                return {"reply": final.choices[0].message.content}

            except json.JSONDecodeError as e:
                logger.error(f"Error parsing tool arguments: {str(e)}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail="Error parsing tool response")
            except Exception as e:
                logger.error(f"Error processing tool call: {str(e)}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Error processing tool call: {str(e)}")

        logger.info("Returning direct response (no tool call)")
        return {"reply": response.choices[0].message.content}

    except HTTPException as he:
        # Re-raise HTTP exceptions as is
        raise he
    except Exception as e:
        logger.error(f"Unhandled error in chat endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
