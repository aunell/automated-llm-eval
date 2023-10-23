import requests
import json
import time
import requests
import json
import re

def create_chat_completion(engine, 
                           system_prompt, 
                           user_prompt, 
                           openai_token, 
                           max_attempts=5,
                           temperature=0.9, 
                           max_tokens=256, 
                           top_p=0.9):
    # set up API key
    headers = {
      'Content-Type': 'application/json',
      'Authorization': f'Bearer {openai_token}'
    }
    data = {
        "model": engine,
        "temperature": temperature,
        "top_p": top_p, 
        "max_tokens": max_tokens, 
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    for attempt in range(max_attempts):
        try:
            print('RESPONSE CONTENT -- MAY BE STUCK')
            response = requests.post('https://api.openai.com/v1/chat/completions', 
                                     headers=headers, 
                                     data=json.dumps(data))
            output_text = response.json()['choices'][0]['message']['content']
            return output_text.strip(), user_prompt
        except Exception as e:
            if attempt < max_attempts - 1:  # i.e. if it's not the final attempt
                sleep_dur = 2 ** attempt  # exponential backoff
                print(f"API call failed with error {e}. Resampling examples and retrying in {sleep_dur} seconds...")
                time.sleep(sleep_dur)
            elif re.search('content management policy', e.user_message):
                return 'ERROR: ' + e.user_message, user_prompt
            else:
                print(f"Prompt {user_prompt} failed after {max_attempts} attempts. Aborting. Error: {e}")
                raise e  # rethrow the last exception if all attempts fail
    return None, None  # If all attempts fail, return None