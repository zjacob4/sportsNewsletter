import os

from autogen import ConversableAgent
import requests

def write_stories(contents):

    cathy = ConversableAgent(
        "cathy",
        system_message="You are a sports writer reporting on the latest news for your newsletter. Your job is to write each story either a headline updates (1-2 sentences), quick summaries (1 paragraph), or deeper dives (3-4 paragraphs). You should also prioritize a good mix of stories from different sports. If you are given feedback by your editor, you should incorporate it and revise your stories. When you get feedback, respond with your revisions but do not address the editor. If you're not given anything to work on, just re-share the stories that were approved. Do not answer your editor with a response, just make the changes to your news story, where applicable.",
        llm_config={"config_list": [{"model": "claude-3-5-sonnet-20240620",
            "api_key": "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",
            "api_type": "anthropic"}]},
        human_input_mode="NEVER",  # Never ask for human input.
    )

    joe = ConversableAgent(
        "joe",
        system_message="You are a sports editor. Your job is to review your writers' work and provide feedback on how to make it more professional, interesting, and grammatically correct. You also ensure that their work reads like a sports newsletter and each story is written as separate snippets.",
        llm_config={"config_list": [{"model": "claude-3-5-sonnet-20240620",
            "api_key": "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",
            "api_type": "anthropic"}]},
        human_input_mode="NEVER",  # Never ask for human input.
    )

    result = joe.initiate_chat(cathy, message=f"Write today's newsletter on the following stories. Use HTML formatting. Here are the stories and how detailed they should be: {contents}", max_turns=2)

    with open("latest_text/latest_news_writing.txt", "w") as file:
        file.write(result.summary)

    return result.summary

if __name__ == '__main__':
    write_stories()