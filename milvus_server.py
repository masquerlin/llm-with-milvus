from pymilvus import DataType, Collection, FieldSchema, CollectionSchema, utility, connections, AnnSearchRequest, WeightedRanker
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
def load_data(files: list[str]):
    documents = []
    for fname in files:
        loader = PyPDFLoader(fname)
        documents += loader.load()


    # 分割文档
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(documents)
    embedding_list = []
    for document in documents:
        document_embedding = model_embedding.encode(document)
        embedding_list.append(document_embedding)
    result = [document, embedding_list]
    return result




sentence_transformer_model_path = ''
model_embedding = SentenceTransformer(sentence_transformer_model_path, device='cpu')
def file_split():
    pass

def update_milvus(files, collection_name):
    data = load_data(files=files)
    connections.connect(uri="http://localhost:19530")
    fields = [
        FieldSchema(name='name', dtype=DataType.VARCHAR, is_primary=True, max_length=18888),
        FieldSchema(name='name_embedding', dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields=fields, enable_dynamic_field=False)
    already = utility.has_collection(collection_name=collection_name)
    if already:
        if utility.has_collection(collection_name=collection_name + 'backup'):
            utility.drop_collection(collection_name=collection_name + 'backup')
    collection = Collection(name=collection_name + 'backup', schema=schema)
    collection.create_index("name", {"index_type":"INVERTED"})
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    collection.create_index("name_embedding", index_params)
    collection.insert(data)
    collection.flush()
    if already:
        utility.drop_collection(collection_name=collection_name)
        utility.rename_collection(old_collection_name=collection_name + 'backup', new_collection_name=collection_name)
    return get_collection()

def get_collection():
    return utility.list_collections()

def get_data_milvus(query, collection_name):
    connections.connect(uri="http://localhost:19530")
    collection = Collection(name=collection_name)
    collection.load()
    query_embedding = [model_embedding.encode(query)]
    search_param_1 = {
        "data": query_embedding, # Query vector
        "anns_field": "name_embedding", # Vector field name
        "param": {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        },
        "limit": 1
    }
    request_1 = AnnSearchRequest(**search_param_1)
    result = collection.hybrid_search(
        reqs=[request_1],
        output_fields=['name'],
        limit=2
        
    )
    collection.release()
    return result


