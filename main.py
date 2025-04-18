from fastapi import FastAPI, Request
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Set this in Render

class ChatRequest(BaseModel):
    message: str
    client_id: str  # for future use

@app.post("/chat")
async def chat(req: ChatRequest):
    user_message = req.message

    # Step 1: Initial GPT call
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful shopping assistant."},
            {"role": "user", "content": user_message}
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

    # Step 2: Tool call?
    if response.choices[0].finish_reason == "tool_calls":
        tool_call = response.choices[0].message.tool_calls[0]
        tool_args = eval(tool_call.function.arguments)
        query = tool_args['query']

        # Step 3: Run the tool (replace with real DB query later)
        fake_results = {
            "products": [
                {"title": "Ashwagandha", "price": "$21.99", "summary": "A calming herb."}
            ]
        }

        # Step 4: Call GPT again with tool result
        final = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful shopping assistant."},
                {"role": "user", "content": user_message},
                {"role": "tool", "name": "recommend_products", "content": str(fake_results)}
            ]
        )

        return { "reply": final.choices[0].message.content }

    return { "reply": response.choices[0].message.content }
