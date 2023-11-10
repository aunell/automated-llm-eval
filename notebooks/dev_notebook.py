# %% [markdown]
# ## Example in how to use ChatModel Wrapper
# %%
import pandas as pd

from automated_llm_eval.chat_model import ChatModel
from automated_llm_eval.utils import ProgressBar

# Instantiate wrapper around OpenAI's API
model = ChatModel(model="gpt-3.5-turbo-1106")
# model = ChatModel(model="gpt-4-1106-preview")
model
# %%
# You can adjust other model settings globally for all API calls
model2 = ChatModel(model="gpt-3.5-turbo-1106", temperature=0.5, top_p=0.5, max_tokens=300, seed=42)
model2
# %% [markdown]
# ### Making API calls using synchronous (blocking) client
# %%
# Make API call, get response message.
# Note: `output_format = "simple"`
response_message = model.create_chat_completion(
    system_message="You are a joke telling machine.",
    user_message="Tell me something about apples.",
    output_format="simple",
)
print(response_message)
# %%
# Make API call, get original ChatCompletion object.
# Note: `output_format = None`
response = model.create_chat_completion(
    system_message="You are a joke telling machine.",
    user_message="Tell me something about apples.",
    output_format=None,
)
print(response)
# %%
# Make API call, get response packaged with input + metadata.
# Note: `output_format = "message_bundle"`
message_bundle = model.create_chat_completion(
    system_message="You are a joke telling machine.",
    user_message="Tell me something about apples.",
    output_format="message_bundle",
)
print(message_bundle)
# %%
# Make API call, get MessageBundle as a dict.
# Note: `output_format = "message_bundle_dict"`
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
# NOTE: if an API call fails, then `None` will be returned. `None` items cannot
# be directly converted into pd.DataFrame
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
# If an API call fails, this method will automatically retry and make another API call.
# By default it will retry 5 times.  We can change this value to 2.
message_bundle_dict = model.create_chat_completion(
    system_message="You are a joke telling machine.",
    user_message="Tell me something about apples.",
    output_format="message_bundle_dict",
    num_retries=2,
)
print(message_bundle_dict)
# %%
# The `create_chat_completion` method is syntactic sugar for `chat_completion`.
# It simply formats the message for us.
system_message = "You are a joke telling machine."
user_message = "Tell me something about apples."
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_message},
]

message_bundle_dict = model.chat_completion(
    messages=messages,
    output_format="message_bundle_dict",
    num_retries=2,
)
print(message_bundle_dict)
# %% [markdown]
# ### Making API calls using asynchronous (non-blocking) client
#
# This enables concurrent API calls.  We can control the max concurrency.
#
# Async uses the asyncio paradigm.  We need to run an asyncio event loop to
# use these functions.
# NOTE: a jupyter notebook has an asyncio event loop running by default,
# but you need to create your own asyncio event loop in a python script
# %%
system_message = "You are a joke telling machine."
user_message = "Tell me something about apples."
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_message},
]

response = await model.async_chat_completion(messages=messages)  # noqa: F704:
response

# %%
# Duplicate Messages x 10 times so that we can make 10 API calls
messages_list = [messages] * 10
messages_list
# %%
# Use Async Chat Completions, limit to 2 concurrent API calls at any given time
responses_list = await model.async_chat_completions(  # noqa: F704
    messages_list=messages_list, num_concurrent=2, output_format="message_bundle_dict"
)

df = pd.DataFrame(responses_list)
df
# %%
