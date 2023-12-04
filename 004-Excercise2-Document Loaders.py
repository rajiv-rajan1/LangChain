# Dependencies 
import os
from langchain.llms import openai
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WikipediaLoader
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


#Load API Key 
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

#Define LLM
chat =ChatOpenAI(open_ai_api_key = api_key)

person_name = input("Enter Person name : ") 
question = input("Enter Question about the person : ")





def answer_question_about(person_name,question):
    docs = WikipediaLoader(query=person_name, load_max_docs=1)
    #data = loader.load()
    data=(docs.load()[0].page_content)
    human_prompt = HumanMessagePromptTemplate.from_template("provide answer about the {person_name} for the Question : {question} using data : {data}")
    chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
    result = chat(chat_prompt.format_prompt().to_messages())
    print(result.content)

answer_question_about(person_name=person_name,question=question)


