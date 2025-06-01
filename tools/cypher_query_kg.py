from langchain_core.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain
from chatbot.llm import llm, cypher_llm
from chatbot.graph import graph



CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher statement to query a graph database.
Instructions:
1.  Use **only** the node labels, relationship types, and property keys provided in the 'Schema:' section below. Do not use any others.
2.  **Node labels are case-sensitive exactly as they appear in the schema.**
3.  **Relationship types are case-sensitive and always uppercase as they appear in the schema.**
4.  When matching string properties (like `name`, `description`, `type`), use case-insensitive matching: `WHERE toLower(node.propertyName) CONTAINS toLower($userInput)`.
5.  If the user provides an exact name for an entity, use exact matching on the relevant property (often `name` or `id` as listed in the schema for that node type): `WHERE node.propertyName = "Exact Value"`.
6.  Return only the properties specifically asked for or those most relevant to providing a concise answer.
7.  For ranking questions ("best", "most popular"), look for properties like `rating` or `averageRating` in the schema and use `ORDER BY ... DESC LIMIT ...`.
8.  Do not return embedding properties.
9.  If the question cannot be answered, respond with "I cannot answer this question with the available graph data."


Examples:

Question: "Which museums are in Alsancak district?"
Cypher:
MATCH (m:Venue)-[:IS_TYPE]->(:Venuetype {id: "Müze"})
MATCH (m)-[:LOCATED_IN]->(d:District {id: "Alsancak"})
RETURN m.id AS museumName


Question: "Deniz Kent Restoran'ı hangi türde  ürünler sunuyor?
Cypher:
MATCH (:Restaurant {id:"Deniz Kent Restoran"})-[:HAS_CUISINE]->(c)  
RETURN c.id  LIMIT 25;

Schema:
{schema}

Question: {question}
Cypher Query:
"""

cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "question"],
)

# create the cypher qa chain
cypher_qa_chain = GraphCypherQAChain.from_llm(
    qa_llm=llm,
    cypher_llm=cypher_llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True,
    exclude_types=["Session", "Message", "LAST_MESSAGE", "NEXT"],
    # enhanced_schema=True,
    allow_dangerous_requests=True
)

def cypher_qa(q):
    return cypher_qa_chain.invoke({"query": q})