import os

from autogen import ConversableAgent
import requests

def add_graphics(contents):

    cathy = ConversableAgent(
        "cathy",
        system_message="You are a graphics designer. Your job is to create graphics for the sports newsletter. Your responsibility is to take HTML code and re-submit the HTML code with links to public domain graphics. You should create a graphic for each story that is visually appealing and relevant to the content. You should also ensure that the graphics are of high quality and are suitable for a professional newsletter. If you are given feedback by your editor, you should incorporate it and revise your graphics. When you get feedback, respond with your revisions but do not address the editor. If you're not given anything to work on, just re-share the graphics that were approved. Do not answer your editor with a response, just make the changes to your graphics, where applicable.",
        llm_config={"config_list": [{"model": "claude-3-5-sonnet-20240620",
            "api_key": "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",
            "api_type": "anthropic"}]},
        human_input_mode="NEVER",  # Never ask for human input.
    )

    joe = ConversableAgent(
        "joe",
        system_message="You are a senior graphics designer. Your job is to review the graphics choices submitted by your junior designer and provide actionable feedback. You should ensure that the graphics are visually appealing, relevant to the content, and of high quality. You should also provide guidance on how to improve the graphics and make them more suitable for a professional newsletter. If you're not given anything to work on, just re-share the graphics that were approved. Do not answer your designer with a response, just make the changes to the graphics, where applicable.",
        llm_config={"config_list": [{"model": "claude-3-5-sonnet-20240620",
            "api_key": "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",
            "api_type": "anthropic"}]},
        human_input_mode="NEVER",  # Never ask for human input.
    )

    result = joe.initiate_chat(cathy, message=f"Add graphics to today's stories and maintain an HTML output: {contents}", max_turns=2
    )

    with open("latest_text/latest_graphics.txt", "w") as file:
        file.write(result.summary)
    return result.summary

if __name__ == '__main__':
    write_stories()