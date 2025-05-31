from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.tools import Tool
from langchain_neo4j import Neo4jChatMessageHistory

from chatbot.utils import get_session_id
from tools.retriever import get_travel_information
from tools.cypher_query_kg import cypher_qa
from chatbot.graph import graph

from chatbot.llm import llm
from langchain.schema import StrOutputParser

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are 'İzmir Gezgini', a friendly and helpful AI travel assistant specializing in providing information and recommendations for visiting İzmir, Turkey. "
         "Engage in general conversation, offer a warm welcome, and guide users if they seem unsure what to ask. "
         "If a question is outside your travel expertise or about locations other than İzmir, politely state that you specialize in İzmir. "
         "Do not answer any questions using your pre-trained knowledge if it's about specific, real-time, or very detailed information that should come from your tools."
         ),
        ("human", "{input}"),
    ]
)

general_chat_chain = chat_prompt | llm | StrOutputParser()

tools = [
    Tool.from_function(
        name="Izmir General Conversation",
        description="Use this for general chit-chat, greetings, or when the user's query is not asking for specific travel information, recommendations, or data from the knowledge graph about İzmir. For example, if the user says 'hello' or 'how are you?'.",
        func=general_chat_chain.invoke,
    ),
    Tool.from_function(
        name="Izmir Semantic Search Travel Information",
        description="Use this tool when a user asks a general question about places, activities, food, or experiences in İzmir that can be answered by searching through descriptive texts and reviews. This is good for open-ended questions like 'What is Konak Square like?' or 'Tell me about seafood restaurants in Alsancak.' or 'What did people say about Kordon Boyu?'. The input should be the user's question about İzmir.",
        func=get_travel_information
    ),
    Tool.from_function(
        name="Izmir Knowledge Graph Cypher Query",
        description="Use this tool when the user asks a specific question about İzmir that requires querying structured data from the knowledge graph. This is useful for questions involving specific attributes, categories, relationships, or comparisons. For example: 'Which museums are in Konak district?', 'What is the entrance fee for Izmir Archeology Museum?', 'List restaurants in Alsancak that offer French cuisine and have an average rating above 4.0.', 'What are the opening hours of Saat Kulesi?'. The input must be the user's direct question.",
        func=cypher_qa,
    )
]

# create chat history callback
def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

#agent prompt
agent_prompt = PromptTemplate.from_template("""
You are 'İzmir Gezgini', an AI travel assistant specializing in providing information about İzmir, Turkey.
Be as helpful and friendly as possible. Your goal is to assist users with their questions about İzmir.

IMPORTANT: Do not answer any questions using your pre-trained knowledge if the question is about specific details, recommendations, or current information for İzmir. You MUST use the provided tools to find information from your knowledge base. If the tools don't provide an answer, state that you don't have the specific information. Your knowledge is limited to İzmir only.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes.  I need to decide which tool is most appropriate based on the user's question about İzmir.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Tool Selection Guide for İzmir:
- For general chit-chat or greetings: Use "Izmir General Conversation".
- For open-ended questions about İzmir (descriptions, what people say, general info): Use "Izmir Semantic Search Travel Information".
- For specific, factual questions about İzmir (attributes, categories, locations, opening hours, specific recommendations based on criteria): Use "Izmir Knowledge Graph Cypher Query".

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")

# create the agent
agent = create_react_agent(llm, tools, agent_prompt)
# create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# create a handler to call the agent
def generate_response(user_input):
    """
        Create a handler that calls the Conversational agent
        and returns a response to be rendered in the UI
    """

    response = chat_agent.invoke(
        {"input":user_input},
        {"configurable": {"session_id": get_session_id()}}
    )

    return response["output"]