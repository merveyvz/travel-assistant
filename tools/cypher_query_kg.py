from langchain_core.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain
from chatbot.llm import llm, cypher_llm
from chatbot.graph import graph


schema = """
Schema:
The graph contains nodes representing districts, places of interest, venues (like restaurants, cafes, museums), 
venue types, cuisine types, specific dishes/products, features of places, user reviews, and persons who wrote reviews.

Available Node Labels and their primary properties:
- `District` (Properties: `name` (string, unique identifier for the district, e.g., "Konak"), `description` (string, optional))
- `PlaceOfInterest` (Properties: `name` (string, unique identifier, e.g., "Saat Kulesi"), `type` (string, e.g., "Tarihi Anıt"), `latitude` (float, optional), `longitude` (float, optional), `addressText` (string, optional), `description` (string, optional), `entranceFee` (string, optional), `openingHours` (string, optional))
- `Venue` (Properties: `name` (string, unique identifier, e.g., "Tarihi Kemeraltı Balıkçısı"), `venueType` (string, e.g., "Restoran", "Müze"), `description` (string, optional), `priceRange` (string, optional), `openingHours` (string, optional), `entranceFee` (string, optional), `contactPhone` (string, optional), `website` (string, optional), `latitude` (float, optional), `longitude` (float, optional), `addressText` (string, optional), `averageRating` (float, optional, calculated or extracted))
- `Restaurant` (This label might be used in conjunction with `Venue` or as a more specific type. Properties are inherited from `Venue` or can be more specific if your LLM extracts them as separate nodes, e.g., `Restaurant {name: 'XYZ'}`)
- `Cafe` (Similar to `Restaurant`)
- `Museum` (Similar to `Restaurant`)
- `HistoricalSite` (Similar to `Restaurant`, often a `PlaceOfInterest` or `Venue`)
- `Park` (Similar to `Restaurant`, often a `PlaceOfInterest` or `Venue`)
- `ShoppingArea` (Similar to `Restaurant`, often a `PlaceOfInterest` or `Venue`)
- `VenueType` (Properties: `name` (string, unique identifier, e.g., "Tarihi Anıt", "Cadde")) - Represents categories.
- `CuisineType` (Properties: `name` (string, unique identifier, e.g., "Deniz Ürünleri", "Türk Mutfağı"))
- `Dish` (Properties: `name` (string, unique identifier, e.g., "İzmir Köfte"), `description` (string, optional))
- `Feature` (Properties: `name` (string, unique identifier, e.g., "Deniz Manzarası"), `description` (string, optional))
- `Review` (Properties: `text` (string), `rating` (float, e.g., 4.5), `reviewerName` (string, optional), `reviewDate` (string or date, optional))
- `Person` (Properties: `name` (string, unique identifier for reviewer)) - Represents the reviewer.
- `Document` (Properties: `id` (string, filename)) - (Usually not directly queried by users)
- `Chunk` (Properties: `id` (string, chunk_id), `text` (string)) - (Usually not directly queried by users for KG questions)

Available Relationship Types and their typical connections:
- `LOCATED_IN` (Connects: `(:PlaceOfInterest|:Venue)-[:LOCATED_IN]->(:District)`, or `(:PlaceOfInterest|:Venue)-[:LOCATED_IN]->(:PlaceOfInterest)` for sub-locations like a shop in a market)
- `IS_TYPE` (Connects: `(:PlaceOfInterest|:Venue)-[:IS_TYPE]->(:VenueType)` )
- `HAS_CUISINE` (Connects: `(:Restaurant|:Cafe)-[:HAS_CUISINE]->(:CuisineType)` )
- `KNOWN_FOR` (Connects: `(:Venue|:PlaceOfInterest)-[:KNOWN_FOR]->(:Dish|:Feature)` )
- `HAS_FEATURE` (Connects: `(:Venue|:PlaceOfInterest)-[:HAS_FEATURE]->(:Feature)` )
- `REVIEWED_BY` (Connects: `(:Review)-[:REVIEWED_BY]->(:Person)` )
- `HAS_REVIEW` (Connects: `(:Venue|:PlaceOfInterest)-[:HAS_REVIEW]->(:Review)` )
- `MENTIONS` (Connects: `(:Review)-[:MENTIONS]->(:PlaceOfInterest|:Venue|:Dish|:Feature)`, or `(:Chunk)-[:MENTIONS]->(various entity nodes)` )
- `NEAR` (Connects: `(:PlaceOfInterest|:Venue)-[:NEAR]-(:PlaceOfInterest|:Venue)`. Properties on relationship: `distanceInMeters` (integer, optional), `travelTimeMinutes` (integer, optional), `distanceText` (string, optional, e.g., "yürüme mesafesinde"))
- `PART_OF` (Connects: `(:Chunk)-[:PART_OF]->(:Document)` ) - (Usually not directly queried by users)
- `HAS_ENTITY` (Connects: `(:Chunk)-[:HAS_ENTITY]->(various entity nodes)` ) - (Usually not directly queried by users)
"""



CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher statement to query a graph database.
Instructions:
1. Use only the node labels and relationship types provided in the schema to construct the Cypher query. Do not use any other node labels or relationship types not explicitly listed.
2. Node labels are case-sensitive as defined in the schema. Relationship types are case-sensitive and always uppercase as defined in the schema.
3. When matching string properties like `name`, `type`, `venueType`, `description`, use case-insensitive matching: `WHERE toLower(node.propertyName) CONTAINS toLower($userInput)`.
4. For properties that are unique identifiers (often `name` for nodes like `District`, `PlaceOfInterest`, `Venue`, `CuisineType`, `Dish`, `Feature`, `Person`), if the user provides an exact name, you can use exact matching: `WHERE node.name = $exactName`.
5. Always try to return a concise set of properties directly relevant to the question. For example, for names, return `node.name`; for descriptions, `node.description`.
6. If the question implies ranking (e.g., "best", "most popular", "highest rated"), use `ORDER BY node.averageRating DESC` (if `Venue.averageRating` exists) or `ORDER BY review.rating DESC`. Always use `DESC` for "best" and `LIMIT` to get top results (e.g., `LIMIT 5`).
7. For "what is X?" or "describe X", retrieve properties of X and information from directly connected descriptive nodes (like `Feature`, `Review`).
8. For locations within a district, use the `District` node and `LOCATED_IN` relationship.
9. For types of places (e.g., "museums in Konak"), you can match on `(:Museum)-[:LOCATED_IN]->(:District {{name: "Konak"}})` if `Museum` is a defined node label, or `(:Venue {{venueType: "Müze"}})-[:LOCATED_IN]->(:District {{name: "Konak"}})` if using a property. Prioritize specific node labels if they exist in the schema.
10. For food/cuisine, use `CuisineType`, `Dish` nodes, and `HAS_CUISINE`, `KNOWN_FOR` relationships.
11. For reviews, use the `Review` node and its properties (`text`, `rating`, `reviewerName`). Connect it to places via `HAS_REVIEW` and to reviewers via `REVIEWED_BY`.
12. Do not return embedding properties like `textEmbedding`.
13. If the question is too complex or requires external knowledge not in the schema, respond with "I cannot answer this question with the available graph data."


Examples (Illustrative - adapt to your exact schema and question, especially node property names like 'name' vs 'id'):

Question: "Which museums are in Konak district?"
Cypher:
MATCH (m:Venue {{venueType: "Müze"}})-[:LOCATED_IN]->(d:District {{name: "Konak"}})
RETURN m.name, m.description, m.entranceFee

Alternatively, if Museum is a distinct node label:
Cypher:
MATCH (m:Museum)-[:LOCATED_IN]->(d:District {{name: "Konak"}})
RETURN m.name, m.description, m.entranceFee

Question: "Who reviewed La Cigale and what did they say?"
Cypher:
MATCH (v:Venue {{name: "La Cigale"}})-[:HAS_REVIEW]->(rev:Review)
OPTIONAL MATCH (rev)-[:REVIEWED_BY]->(p:Person)
RETURN p.name AS reviewerName, rev.text AS reviewText, rev.rating

Question: "What restaurants in Alsancak offer seafood?"
Cypher:
MATCH (r:Restaurant)-[:LOCATED_IN]->(d:District {{name: "Alsancak"}})
MATCH (r)-[:HAS_CUISINE]->(ct:CuisineType {{name: "Deniz Ürünleri"}})
RETURN r.name, r.priceRange, r.averageRating

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