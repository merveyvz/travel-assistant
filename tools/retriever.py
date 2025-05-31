from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jVector

from chatbot.llm import llm, embedding_provider
from chatbot.graph import graph

chunk_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    graph=graph,
    index_name="chunkVector",
    embedding_node_property="textEmbedding",
    text_node_property="text",
    retrieval_query="""
        // get the document
        MATCH (node)-[:PART_OF]->(d:Document)
        WITH node, score, d
        
        // get the entities and relationships for the document
        MATCH (node)-[:HAS_ENTITY]->(e)
        MATCH p = (e)-[r]-(e2)
        WHERE (node)-[:HAS_ENTITY]->(e2)
        
        // unwind the path, create a string of the entities and relationships
        UNWIND relationships(p) as rels
        WITH 
            node, 
            score, 
            d, 
            collect(apoc.text.join(
                [labels(startNode(rels))[0], startNode(rels).id, type(rels), labels(endNode(rels))[0], endNode(rels).id]
                ," ")) as kg
        RETURN
            node.text as text, score,
            { 
                document: d.id,
                entities: kg
            } AS metadata
        """
)


instructions = (
    "Use the given context to answer the question."
    "Reply with an answer that includes the id of the document and other relevant information from the text."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)


chunk_retriever = chunk_vector.as_retriever()
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chunk_retriever = create_retrieval_chain(
    chunk_retriever,
    question_answer_chain
)

def get_travel_information(q):
    return chunk_retriever.invoke({"input": q})
