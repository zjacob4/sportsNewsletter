from openai import OpenAI

YOUR_API_KEY = "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"

messages = [
    {
        "role": "system",
        "content": (
            "You are an artificial intelligence assistant and you need to "
            "engage in a helpful, detailed, polite conversation with a user."
        ),
    },
    {
        "role": "user",
        "content": (
            "You are a sports news researcher. Give me a dozen news stories that are a good balance of new and important. Ensure a mix across various sports, with more focus on more popular sports."
        ),
    },
]

client = OpenAI(api_key=YOUR_API_KEY, base_url="https://api.perplexity.ai")

# demo chat completion without streaming
response = client.chat.completions.create(
    model="llama-3.1-sonar-small-128k-online",
    messages=messages,
)
print(response)

# demo chat completion with streaming
#response_stream = client.chat.completions.create(
#    model="mistral-7b-instruct",
#    messages=messages,
#    stream=True,
#)
#for response in response_stream:
#    print(response)

#perplexity_api_key = "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"