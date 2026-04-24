# %%
from openai import OpenAI
import os

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Local PDF path
pdf_path = r"/path/to/your/file.pdf"   # <-- change this

# Step 1: Upload file
with open(pdf_path, "rb") as f:
    uploaded_file = client.files.create(
        file=f,
        purpose="user_data"
    )

print("Uploaded File ID:", uploaded_file.id)

# Step 2: Ask question using uploaded file
response = client.responses.create(
    model="gpt-5",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "file_id": uploaded_file.id
                },
                {
                    "type": "input_text",
                    "text": "Summarize this PDF and extract the key points."
                }
            ]
        }
    ]
)

# Step 3: Output response text
print("\n=== Model Response ===")
print(response.output_text)

# Step 4: Token usage
print("\n=== Token Usage ===")
print("Input Tokens:", response.usage.input_tokens)
print("Output Tokens:", response.usage.output_tokens)
print("Total Tokens:", response.usage.total_tokens)