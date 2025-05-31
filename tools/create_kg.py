from chatbot.llm import llm, embedding_provider
from chatbot.graph import graph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain_community.graphs.graph_document import Node, Relationship

DOCS_PATH = "/data"

doc_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=[
        "District",         # İlçe (Konak, Alsancak)
        "PlaceOfInterest",  # Gezilecek Yer (Saat Kulesi, Kordon Boyu)
        "Venue",            # Genel Mekan (Restoran, Kafe, Mağaza, Müze vb. için üst tip)
        "Restaurant",       # Yeme İçme Mekanı (özellikle restoranlar)
        "Cafe",             # Kafeler
        "VenueType",        # Mekan Türü (Tarihi Anıt, Cadde, Sahil Şeridi, Müze)
        "CuisineType",      # Mutfak Türü (Deniz Ürünleri, Türk Mutfağı, Fransız Mutfağı)
        "Dish",             # Özel Yemek/Ürün (su böreği, sakızlı dondurma)
        "Feature",          # Öne Çıkan Özellik (tarihi doku, deniz manzarası, canlı atmosfer)
        "Review",           # İnceleme/Yorum (bir bütün olarak yorum metni ve puanı)
        "Person"            # İncelemeyi yapan kişi
    ],
    allowed_relationships=[
        "LOCATED_IN",       # Bir yerin bir ilçede veya daha büyük bir konumda olması
        "IS_TYPE",          # Bir yerin türünü belirtir (Saat Kulesi IS_TYPE Tarihi Anıt)
        "HAS_CUISINE",      # Bir restoranın mutfak türü
        "KNOWN_FOR",        # Bir mekanın bilinen bir yemeği veya özelliği
        "HAS_FEATURE",      # Bir yerin sahip olduğu özellik
        "REVIEWED_BY",      # Bir yerin kim tarafından incelendiği
        "HAS_REVIEW",       # Bir yerin bir incelemesi olması
        "MENTIONS",         # Bir incelemenin bir yerden veya özellikten bahsetmesi
        "NEAR"              # İki mekanın birbirine yakın olması (Konum İpucu'ndan çıkarılabilir)
    ]
)

# load and split the documents
def load_and_split_documents():
    loader = DirectoryLoader(DOCS_PATH, glob="**/*.txt", loader_cls=TextLoader,loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()

    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1500,
        chunk_overlap=200,
    )

    return text_splitter.split_documents(docs)

chunks = load_and_split_documents()

# Dosya bazında chunk'ları numaralandırmak için bir sözlük
chunk_counters = {}

for chunk in chunks:
    filename = os.path.basename(chunk.metadata["source"])
    # Bu dosya için chunk sayacını al veya başlat
    if filename not in chunk_counters:
        chunk_counters[filename] = 0
    chunk_counters[filename] += 1
    chunk_index = chunk_counters[filename]

    # chunk_id'yi dosya adı ve chunk sırasıyla oluştur
    chunk_id = f"{filename}_chunk_{chunk_index}"  # Örnek: "konak_izmir.txt_chunk_1"
    print("Processing -", chunk_id)

    # Embed the chunk
    chunk_embedding = embedding_provider.embed_query(chunk.page_content)

    # Add the Document and Chunk nodes to the graph
    properties = {
        "filename": filename,
        "chunk_id": chunk_id,
        "text": chunk.page_content,
        "embedding": chunk_embedding
    }

    graph.query("""
        MERGE (d:Document {id: $filename})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text
        MERGE (d)<-[:PART_OF]-(c)
        WITH c
        CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
        """,
                properties
                )

    # Generate the entities and relationships from the chunk
    graph_docs = doc_transformer.convert_to_graph_documents([chunk])

    # Map the entities in the graph documents to the chunk node
    for graph_doc in graph_docs:
        chunk_node = Node(
            id=chunk_id,
            type="Chunk"
        )

        for node in graph_doc.nodes:
            graph_doc.relationships.append(
                Relationship(
                    source=chunk_node,
                    target=node,
                    type="HAS_ENTITY"
                )
            )

        # add the graph documents to the graph
    graph.add_graph_documents(graph_docs)

# Create the vector index
graph.query("""
    CREATE VECTOR INDEX `chunkVector`
    IF NOT EXISTS
    FOR (c: Chunk) ON (c.textEmbedding)
    OPTIONS {indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
    }};""")

