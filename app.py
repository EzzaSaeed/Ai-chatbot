import os
import gradio as gr
import spaces
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

SYSTEM_PROMPT = """
You are Neurofive Solutions Support Assistant.
Rules:
- Introduce yourself as Neurofive Support.
- Be friendly and professional.
- Help only with AI, Prompt Engineering, Python, APIs, Hugging Face, and Gradio.
- If asked unrelated questions, politely say that your role is limited to technical support.
- Stay in character.
"""

@spaces.GPU
def chatbot(message, history):
    try:
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        # Works with Gradio 5
        if history:
            for item in history:
                if isinstance(item, dict):
                    messages.append(
                        {
                            "role": item["role"],
                            "content": item["content"]
                        }
                    )

        messages.append(
            {
                "role": "user",
                "content": message
            }
        )

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b:free",
            messages=messages
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {e}"

demo = gr.ChatInterface(
    fn=chatbot,
    title="🤖 Neurofive Solutions Support Assistant",
    description="Powered by OpenRouter"
)

demo.launch()
