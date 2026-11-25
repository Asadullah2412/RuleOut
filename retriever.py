from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-en-v1.5",
    model_kwargs = {"device":"cpu"},
    encode_kwargs ={"normalize_embeddings": True}

)





db = Chroma(
    collection_name= "RuleOut",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)



def top_retriever(keywords , top_k=3):
    """
    Searches for the keywords and returns the top 3 (default 3) with meta data
    """

    content = db.similarity_search(
        keywords,
        k=top_k
    )

    response = {
        "content" : {},
        "metadata" : {}
    }
  
    for i , result in enumerate(content,1):

        response["content"][f'Result {i}'] = result.page_content
        
        response['metadata'][f'metadata{i}'] = result.metadata
    
    return response





# query =  "What are the clinical features of bronchiolitis in infants?"


# result = top_retriever(query)
# print(result)

# results = db.similarity_search(
#     query,
#     k=5
# )
# print("Question >>>> ",query)
# for i , result in enumerate(results,1):
#     print(f"\n{'=' * 60}")
#     print(f"RESULT {i}")
#     print(f"{'=' * 60}")

#     print(result.page_content)
#     print(result.metadata)