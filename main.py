from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
import traceback
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow frontend to access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can lock this down to your frontend domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def log_openai_response(response: Dict[Any, Any], step: str) -> None:
    """Helper function to log OpenAI response details"""
    logger.info(f"OpenAI Response ({step}):")
    logger.info(f"Finish Reason: {response.choices[0].finish_reason}")
    logger.info(f"Model Used: {response.model}")
    logger.info(f"Usage: {response.usage}")

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        logger.info(f"Received chat request from client {req.client_id}")
        logger.info(f"User message: {req.message}")
        
        # Step 1: Initial GPT call
        logger.info("Making initial GPT call...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
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
                }]
            )
            log_openai_response(response, "initial")
            
        except Exception as e:
            logger.error(f"Error in initial GPT call: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

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
                        model="gpt-4o",
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
                        ]
                    )
                    log_openai_response(final, "final")
                    
                except Exception as e:
                    logger.error(f"Error in final GPT call: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise HTTPException(status_code=500, detail=f"OpenAI API error in final call: {str(e)}")

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

    except Exception as e:
        logger.error(f"Unhandled error in chat endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
