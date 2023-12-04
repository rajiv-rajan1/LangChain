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

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

#STEP1 : Create a System Template and System Message prompt

system_template = "You are a AI trainer who can create training prompts from existing data, create minimum 5 question and answer pair and output both human question and AI Answer"
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

#STEP2 : Create a input and output example, where Legal text is the input and simplified text is output. 
# these these are added to Human and AI message prompt Template. 
legal_text = """
Symptoms:
In a VMware SD-WAN Edge:

    BGP does not come up.
    Edge fails to establish a BGP session with devices connected to non-global segment
    The following conditions are true:
        BGP is configured in both the global segment and a non-global segment.
        In the non-global segment, BGP is configured on a subinterface.
        The same IP address is configured in different segments (for example, the same IP is used in a WAN overlay interface on the global segment, and in a subinterface of the non-global segment). 

The issue can happen after a reboot of software upgrade.
Cause:
The issue is documented in bug #102655.
The issue happens because the Edge sends the BGP packets on the non-global segment without a VLAN tag, eventually causing the BGP peer to send a reset on the TCP connection.

Resolution:
This issue is resolved in SD-WAN Edge versions 4.5.2 and 5.2.0
For information on how to upgrade please check the following article: VMware SD-WAN Software Upgrade FAQs


"""


example_input_one = HumanMessagePromptTemplate.from_template(legal_text)
plain_text = """
human: What are the known issues with BGP in VMWARE SDWAN?
AI: under certain conditions, Edge fails to establish a BGP session with devices connected to non-global segment. refer to bug #102655. 
human: What is the details of bug id 102655?
AI: under certain conditions, Edge fails to establish a BGP session with devices connected to non-global segment. refer to bug #102655.
human: Why BGP fails on non global segments after reboot ?
AI: under certain conditions, Edge fails to establish a BGP session with devices connected to non-global segment. refer to bug #102655.
human: Why BGP fails on non global segments after upgrade?
AI: under certain conditions, Edge fails to establish a BGP session with devices connected to non-global segment. refer to bug #102655.
"""
example_output_one = AIMessagePromptTemplate.from_template(plain_text)

#STEP3 : add the enitre Legal text as human message prompt. 
human_template = "{legal_text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

#STEP4: Construct the Chat prompt
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, example_input_one, example_output_one, human_message_prompt]
)
#STEP5 : using another exampample and sending the chat prompt and new text to chat model. 
some_example_text = """
Symptoms:
In a VMWARE EDGE, During testing of software releases, it was discovered when upgrading a High Availability pair of edges the upgrade may not complete and will not retry automatically.

Cause:
The issue is being tracked under ID #105160.
The issue causes an exception in the Edge upgrade process which prevents the edge from upgrading as it assumes the upgrade was completed.
Resolution:
This issue was resolved in these GA releases, SD-WAN Edge version 5.0.1.4 build R5014-20230713-GA (5.0.1 Release Notes), and in 5.2.0.0 (5.2.0 Release Notes )
 """
request = chat_prompt.format_prompt(legal_text=some_example_text).to_messages()


result = chat(request)

print(result.content)