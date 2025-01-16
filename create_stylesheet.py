import os

from autogen import ConversableAgent
import requests

def write_css_stylesheet(html_script):

    cathy = ConversableAgent(
        "cathy",
        system_message="You are a CSS coder. Your job is to write a CSS stylesheet for a sports newsletter. Your stylesheet should include styles for the elements in the HTML script you're given. You should also include a media query for mobile devices. Only ever respond with the CSS stylesheet. Do not respond to the senior coder, just respond with the updated CSS stylesheet.",
        llm_config={"config_list": [{"model": "claude-3-5-sonnet-20240620",
            "api_key": "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",
            "api_type": "anthropic"}]},
        human_input_mode="NEVER",  # Never ask for human input.
    )

    joe = ConversableAgent(
        "joe",
        system_message="You are a senior CSS coder. Your job is to review the CSS stylesheet written by your junior coder and provide feedback on how to make it more professional, efficient, and responsive. You also ensure that the stylesheet is well-organized and follows best practices.",
        llm_config={"config_list": [{"model": "claude-3-5-sonnet-20240620",
            "api_key": "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",
            "api_type": "anthropic"}]},
        human_input_mode="NEVER",  # Never ask for human input.
    )

    result = joe.initiate_chat(cathy, message=f"Write today's newsletter's CSS stylesheet for the following HTML script: {html_script}", max_turns=3)
    create_stylesheet(result.summary)
    
    return None


def create_stylesheet(css):
    with open("static/styles.css", "w") as f:
        f.write(css)
    return None

if __name__ == '__main__':
    write_stories()