import os

from autogen import ConversableAgent
import requests
from openai import OpenAI

def fact_check_stories(contents):

    perplexity_API_key = "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"

    response = requests.get("https://newsapi.org/v2/top-headlines?country=us&category=sports&apiKey=21c7c1507cb3456fb95ae64cc45c0180")
    if response.status_code == 200:
        data = response.json()
        descriptions = [article['description'] for article in data['articles']]
        news = [article['content'] for article in data['articles']]
        # Process the descriptions here
    else:
        print("Error: Failed to retrieve data from the API")

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
                f"You are a fact-checker for a sports news outlet. Review the following stories and repeat the stories back, exactly as you received them, replacing any incorrect information with the correct facts: {contents} . Read the following news as well and use this to fact check (start here and then use the internet for anything not provided here):  {news} "
            ),
        },
    ]

    client = OpenAI(api_key=perplexity_API_key, base_url="https://api.perplexity.ai")

    # demo chat completion without streaming
    background = client.chat.completions.create(
        model="llama-3.1-sonar-small-128k-online",
        messages=messages,
    )

    # Extract the text content from the ChatCompletion object
    fact_checked_text = background.choices[0].message.content

    with open("latest_text/latest_fact_check.txt", "w") as file:
        file.write(fact_checked_text)

    #return stories_and_research
    return fact_checked_text

if __name__ == '__main__':
    fact_check_stories()