import os
from autogen import ConversableAgent

# Set the environment variable within the script
#os.environ['OPENAI_API_KEY'] = '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'

agent = ConversableAgent(
    "chatbot",
    llm_config={"config_list": [{"model": "claude-3-5-sonnet-20240620",
        "api_key": "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",
        "api_type": "anthropic"}]},
    code_execution_config=False,  # Turn off code execution, by default it is off.
    function_map=None,  # No registered functions, by default it is None.
    human_input_mode="NEVER",  # Never ask for human input.
)

reply = agent.generate_reply(messages=[{"content": "Tell me a joke.", "role": "user"}])
print(reply)