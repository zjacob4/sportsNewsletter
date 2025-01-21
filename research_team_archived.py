import os

from autogen import ConversableAgent
import requests

def research_stories():
    response = requests.get("https://newsapi.org/v2/top-headlines?country=us&category=sports&apiKey=21c7c1507cb3456fb95ae64cc45c0180")
    if response.status_code == 200:
        data = response.json()
        descriptions = [article['description'] for article in data['articles']]
        contents = [article['content'] for article in data['articles']]
        # Process the descriptions here
    else:
        print("Error: Failed to retrieve data from the API")


    cathy = ConversableAgent(
        "cathy",
        system_message="You are a sports news researcher. Your job is to take the latest sports news provided by your senior researcher and relay the most interesting and important stories to your senior researcher, who will give you feedback on how to prioritize the most important stories and what additional background to provide. You will present stories as candidates for either headline updates (1-2 sentences), quick summaries (1 paragraph), and deeper dives (3-4 paragraphs). Never make up details. If your senior approves a story for a deeper dive, search the internet for more context/background on the story.",
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

    result = joe.initiate_chat(cathy, message=f"Provide today's news stories and whether they should be headlines, summaries, or deep dives. There should be a good mix of each type. Here is the content you can use: {contents}", max_turns=3)
    return result.summary

if __name__ == '__main__':
    research_stories()