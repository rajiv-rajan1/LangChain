#Dependencies 
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate


# An example prompt with no input variables
sample_input_prompt = PromptTemplate(input_variables=['topic',"level"], 
                                     template="Tell me a fact about {topic} for a student, studying {level} level")
request = sample_input_prompt.format(topic='mars',level='phd')


# Loading API Key 
load_dotenv()
open_ai_api_key = os.getenv('OPENAI_API_KEY')


#Code 
llm = OpenAI(openai_api_key=open_ai_api_key)
#print(llm('Here is a fun fact about Pluto:'))
#result = llm.generate(['Here is a fun fact about Pluto:','Here is a fun fact about Mars:'])
#print(result.schema)

prompt_Test = llm(request)
print(prompt_Test)