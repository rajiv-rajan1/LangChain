# TASK : Create a python function that uses Prompts and Chat internally to give travel ideas related to two variables.
#       1. An interest or hubby 
#       2. A budget 

#Dependencies 
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

# Loading API Key 
load_dotenv()
open_ai_api_key = os.getenv('OPENAI_API_KEY')

#define LLM 
from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(openai_api_key=open_ai_api_key)

#STEP1 : Importing Lanchain templates 

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

def my_fun():

    #STEP2 : use System message promt Template and constructing system message 
    system_template="You are an AI Travel assistant, who can guide based on {interest_or_hubby} and for a {budget}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    print(system_message_prompt.input_variables)

    #STEP3 : use Human message prompt tempalte and constructing human message 
    human_template="{travel_guide_request}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    print(human_message_prompt.input_variables)

    #STEP4 : use chatprmpt Template and constructing chat prompt with the above two. 
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    print(chat_prompt.input_variables)

    #STEP 5 : Construct and rquest using chat prompt and format it
    request = chat_prompt.format_prompt(interest_or_hubby="hiking", budget="$1000", travel_guide_request="Want to Travel with in india").to_messages()
    print(request)

    # STEP 6 : Send request to LLM 
    result = chat(request)
    print(result.content)
my_fun()