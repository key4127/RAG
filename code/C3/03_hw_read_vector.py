from llama_index.core import StorageContext, Settings, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = None
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

persist_path = "./llamaindex_index_store"

storage_context = StorageContext.from_defaults(persist_dir=persist_path)

index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()
response = query_engine.query("LlamaIndex是什么？")
print("相似度最高的文档: \n")
print(response)