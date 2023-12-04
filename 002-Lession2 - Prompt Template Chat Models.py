#Dependencies 
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

# Loading API Key 
load_dotenv()
open_ai_api_key = os.getenv('OPENAI_API_KEY')

#STEP1 : Importing open ai templates 

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

#define LLM 
from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(openai_api_key=open_ai_api_key)

#STEP2 : use System message promt Template and constructing system message 
system_template="You are an AI recipe assistant that specializes in {dietary_preference} dishes that can be prepared in {cooking_time}."
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
print(system_message_prompt.input_variables)

#STEP3 : use Human message prompt tempalte and constructing human message 
human_template="{recipe_request}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
print(human_message_prompt.input_variables)

#STEP4 : use chatprmpt Template and constructing chat prompt with the above two. 
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
print(chat_prompt.input_variables)

#STEP 5 : Construct and rquest using chat prompt and format it
request = chat_prompt.format_prompt(cooking_time="15 min", dietary_preference="Vegan", recipe_request="Quick Snack").to_messages()
print(request)


# STEP 6 : Send request to LLM 
result = chat(request)
print(result.content)