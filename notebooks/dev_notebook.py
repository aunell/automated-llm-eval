# %%
import pandas as pd

from automated_llm_eval.chat_model import ChatModel
from automated_llm_eval.utils import ProgressBar

# Instantiate wrapper around OpenAI's API
model = ChatModel(model="gpt-3.5-turbo-1106")
# model = ChatModel(model="gpt-4-1106-preview")
# %%
# Make API call, get response message.  Note: `output_format = "simple"`
response_message = model.create_chat_completion(
    system_message="You are a joke telling machine.",
    user_message="Tell me something about apples.",
    output_format="simple",
)
print(response_message)
# %%
# Make API call, get original ChatCompletion object.  Note: `output_format = None`
response = model.create_chat_completion(
    system_message="You are a joke telling machine.",
    user_message="Tell me something about apples.",
    output_format=None,
)
print(response)
# %%
# Make API call, get response packaged with input + metadata.  Note: `output_format = "message_bundle"`
message_bundle = model.create_chat_completion(
    system_message="You are a joke telling machine.",
    user_message="Tell me something about apples.",
    output_format="message_bundle",
)
print(message_bundle)
# %%
# Make API call, get MessageBundle as a dict.  Note: `output_format = "message_bundle_dict"`
message_bundle_dict = model.create_chat_completion(
    system_message="You are a joke telling machine.",
    user_message="Tell me something about apples.",
    output_format="message_bundle_dict",
)
print(message_bundle_dict)
# %%
# Message bundle dict can be converted into pandas Series easily
s = pd.Series(message_bundle_dict)
s
# %%
# Multiple message bundle dicts can be converted into pandas DataFrame
responses = []
with ProgressBar() as p:
    for _ in p.track(range(5)):
        response = model.create_chat_completion(
            system_message="You are a joke telling machine.",
            user_message="Tell me something about apples.",
            output_format="message_bundle_dict",
            temperature=0.4,
            seed=None,
        )
        responses += [response]

df = pd.DataFrame(responses)
df
# %%

# %%
