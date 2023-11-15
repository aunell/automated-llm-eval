# %% [markdown]
# ## Example in how to use ChatModel Wrapper
# %%
import pandas as pd

from automated_llm_eval.chat_model import ChatModel, Message
from automated_llm_eval.utils import ProgressBar, sidethread_event_loop_async_runner

# Instantiate wrapper around OpenAI's API
model = ChatModel(model="gpt-3.5-turbo-1106")
# model = ChatModel(model="gpt-4-1106-preview")
model
# %%
# You can adjust other model settings globally for all API calls
model2 = ChatModel(model="gpt-3.5-turbo-1106", temperature=0.5, top_p=0.5, max_tokens=300, seed=42)
model2
# %%
# `max_tokens = None` means no max_token limit (this is the default)
model2 = ChatModel(model="gpt-3.5-turbo-1106", temperature=0.5, top_p=0.5, max_tokens=None, seed=42)
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
# Note: `output_format = "bundle"`
bundle = model.create_chat_completion(
    system_message="You are a joke telling machine.",
    user_message="Tell me something about apples.",
    output_format="bundle",
)
print(bundle)
# %%
# Make API call, get MessageBundle as a dict.
# Note: `output_format = "bundle_dict"`
bundle_dict = model.create_chat_completion(
    system_message="You are a joke telling machine.",
    user_message="Tell me something about apples.",
    output_format="bundle_dict",
)
print(bundle_dict)
# %%
# Message bundle dict can be converted into pandas Series easily
s = pd.Series(bundle_dict)
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
            output_format="bundle_dict",
            temperature=0.4,
            seed=None,
        )
        responses += [response]

df = pd.DataFrame(responses)
df
# %%
# If an API call fails, this method will automatically retry and make another API call.
# By default it will retry 5 times.  We can change this value to 2.
bundle_dict = model.create_chat_completion(
    system_message="You are a joke telling machine.",
    user_message="Tell me something about apples.",
    output_format="bundle_dict",
    num_retries=2,
)
print(bundle_dict)
# %%
# The `create_chat_completion` method is syntactic sugar for `chat_completion`.
# It simply formats the message for us.
system_message = "You are a joke telling machine."
user_message = "Tell me something about apples."
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_message},
]

bundle_dict = model.chat_completion(
    messages=messages,
    output_format="bundle_dict",
    num_retries=2,
)
print(bundle_dict)
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

response = await model.async_chat_completion(messages=messages, num_retries=1)  # noqa: F704:
response

# %%
# Duplicate Messages x 5 times so that we can make 5 API calls
messages_list = [messages] * 5
messages_list
# %%
# Use Async Chat Completions, limit to 2 concurrent API calls at any given time
responses_list = await model.async_chat_completions(  # noqa: F704
    messages_list=messages_list,
    num_concurrent=2,
    num_retries=1,
    output_format="bundle_dict",
)

df = pd.DataFrame(responses_list)
df
# %% [markdown]
# ### Example of using `Message` and `validation_callback`
#
# The `Message` wrapper allows packaging arbitrary user-defined metadata along with each message
# which is a good place to put labels, notes, etc.
#
# The `validation_callback` argument enables the user to define
# specific logic to validate the response from each API call to OpenAI
# for each message.  Passed into the callback function is the original
# `messages` and the `response`.  If the `messages` is a `Message` object,
# this will be returned in `validation_callback` for access to all metadata.
# `response` is the LLM response after being parsed and formated as specified
# in `output_format`.

# %%
system_message = "You are a joke telling machine."
user_message = "Tell me something about apples."
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_message},
]
m = Message(messages=messages, metadata={"a": 1})


def validation_callback_fn(messages, response) -> bool:
    print(f"In Callback. Messages: {messages}")
    print(f"In Callback. Response: {response}")
    print("\n")
    metadata = messages.metadata
    if "a" in metadata:
        return metadata["a"] == 1
    else:
        return False


# Instantiate wrapper around OpenAI's API
model = ChatModel(model="gpt-3.5-turbo-1106")
# Make ChatCompletion with...
# - using Message wrapper and include metadata (ChatModel automatically unpacks Message.messages)
# - parse raw OpenAI response into "simple" string format
# - then call the `validation_callback_fn` that we defined.  ChatModel always passes in
#   original messages input and parsed response as the 1st and 2nd arguments.  The
#   `validation_callback_fn` can contain any logic, but ultimately needs to return `True` vs `False`
#   to accept or reject the response.  If the response is rejected, ChatModel automatically retries.
# - allow up to 1 retry.  If still fails/rejected after 1 retry, then will return `None`.
response = model.chat_completion(
    m,
    output_format="bundle_dict",
    validation_callback=validation_callback_fn,
    num_retries=1,
)
response

# %%
# Multiple concurrent async chat completions using Message
# NOTE: we make the 3rd Message with different metadata.  This should cause
# the `validation_callback_fn` to reject the response for only the 3rd Message in list
# and retry only the 3rd Message.
msg_list = [m] * 2 + [Message(messages=messages, metadata={"b": 2})]
msg_list
# %%
# Use Async Chat Completions, limit to 2 concurrent API calls at any given time & 1 retry
responses_list = await model.async_chat_completions(  # noqa: F704
    messages_list=msg_list,
    num_concurrent=2,
    num_retries=1,
    validation_callback=validation_callback_fn,
    output_format="bundle_dict",
)
# %%
# Examine responses.
# - We should get valid responses for the first 2 responses.
# - The 3rd response should always be `None` because the metadata cannot pass at
#   `validation_callback_fn`
responses_list

# %% [markdown]
# ### Calling Async function from Sync code
# %%
model = ChatModel(model="gpt-3.5-turbo-1106")

system_message = "You are a joke telling machine."
user_message = "Tell me something about apples."
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_message},
]
m = Message(messages=messages, metadata={"a": 1})
msg_list = [m] * 3

# %%
# Up until now, we have used `await` to call async functions and wait for their completion.
# However, `await` this can only be used within async functions.
# we are not allowed to call `await` from a function not defined with `async def`
responses = await model.async_chat_completions(
    messages_list=msg_list, num_concurrent=2, output_format="bundle"
)
responses

# %%
# We have created a helper function to address this issue.
#
# Call async method from sync function without using `await` keyword.
# This involves creating an event loop on another thread, then
# waiting for result on main thread and shutting down the event loop on other thread.

result = sidethread_event_loop_async_runner(
    async_function=model.async_chat_completions(
        messages_list=msg_list, num_concurrent=2, output_format="bundle"
    )
)
result

# %%
