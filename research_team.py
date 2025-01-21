import os

from autogen import ConversableAgent
import requests
from openai import OpenAI

def research_stories():

    response = requests.get("https://newsapi.org/v2/top-headlines?country=us&category=sports&apiKey=21c7c1507cb3456fb95ae64cc45c0180")
    if response.status_code == 200:
        data = response.json()
        descriptions = [article['description'] for article in data['articles']]
        contents = [article['content'] for article in data['articles']]
        # Process the descriptions here
    else:
        print("Error: Failed to retrieve data from the API")

    with open("latest_text/latest_stories.txt", "w") as file:
        for content in contents:
            if content:  # Check if content is not None
                file.write(content + "\n")


    cathy = ConversableAgent(
        "cathy",
        system_message="You are a sports news researcher. Your job is to browse the latest sports news given to you by your senior researcher and relay the most interesting and important stories to your senior researcher, who will give you feedback on how to prioritize the most important stories and what additional background to provide. You will present stories as candidates for either headline updates (1-2 sentences), quick summaries (1 paragraph), and deeper dives (3-4 paragraphs). Never make up details and only give facts that your senior researcher gives to you.",
        llm_config={"config_list": [{"model": "claude-3-5-sonnet-20240620",
            "api_key": "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",
            "api_type": "anthropic"}]},
        human_input_mode="NEVER",  # Never ask for human input.
    )

    joe = ConversableAgent(
        "joe",
        system_message="You are a senior sports news researcher. Your job is to review stories presented to you by your junior editor and provide feedback on how to prioritize the most important stories (based on relevance, significance, and likely public interest) and what additional background to provide. You will also provide feedback on the quality of the stories and suggest improvements. You will suggest stories as candidates for either headline updates (1-2 sentences), quick summaries (1 paragraph), and deeper dives (3-4 paragraphs).",
        llm_config={"config_list": [{"model": "claude-3-5-sonnet-20240620",
            "api_key": "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",
            "api_type": "anthropic"}]},
        human_input_mode="NEVER",  # Never ask for human input.
    )

    result = joe.initiate_chat(cathy, message=f"I am your senior researcher. Use these news stories and determine whether they should be headlines, summaries, or deep dives. There should be a good mix of each type. Here is the content you can use: {contents}", max_turns=2)

    news_stories = result.summary

    perplexity_API_key = "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"

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
                f"You are a sports news researcher. Take the following news stories and, for the ones that are assigned as deep dives, provide a few paragraphs of additional context/background on the story. For the ones that are assigned as quick summaries, provide a single paragraph of additional context/background. For the ones that are assigned as headline updates, provide a single sentence of additional context/background. Here are the news stories: {news_stories} . If any of the stories seem ambiguous, here is the full details where those stories are pulled from: {contents}"
            ),
        },
    ]

    with open("latest_text/what_perplexity_sees.txt", "w") as file:
        file.write(
            f"You are a sports news researcher. Take the following news stories and, for the ones that are assigned as deep dives, provide a few paragraphs of additional context/background on the story. For the ones that are assigned as quick summaries, provide a single paragraph of additional context/background. For the ones that are assigned as headline updates, provide a single sentence of additional context/background. Here are the news stories: {news_stories} . If any of the stories seem ambiguous, here is the full details where those stories are pulled from: {contents}"
        )

    client = OpenAI(api_key=perplexity_API_key, base_url="https://api.perplexity.ai")

    # demo chat completion without streaming
    background = client.chat.completions.create(
        model="llama-3.1-sonar-small-128k-online",
        messages=messages,
    )

    # Extract the text content from the ChatCompletion object
    background_text = background.choices[0].message.content
    
    stories_and_research = result.summary + background_text

    with open("latest_text/latest_story_selection.txt", "w") as file:
        file.write(result.summary)

    with open("latest_text/latest_research.txt", "w") as file:
        file.write(background_text)

    #return stories_and_research
    return background_text

if __name__ == '__main__':
    research_stories()