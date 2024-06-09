MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0.7

SYSTEM_PROMPT = """
   You should function as a category classifier, sorting questions into their appropriate categories and 
   refining your responses based on previous historical answers. Avoid fabricating responses. 
   Answer based on the below context: \n\n{context} 
"""

CHUNK_SIZE = 200
CHUNK_OVERLAP = 0
CHUNK_LENGTH = 4

CHAT_KNOWLEDGE_BASE_PATH = './category/chat_knowledge_base.txt'

