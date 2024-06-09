import os

from decouple import config

from category import constants
from category import enum

from rest_framework import serializers

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.document_loaders import TextLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChatOpenAIUtils(object):
    os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')

    def __init__(self, prompt, model_name=constants.MODEL_NAME, temperature=constants.TEMPERATURE):
        self.prompt = prompt
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

    def get_response(self):
        if self.prompt:
            ChatHistoryUtils.add_message(self.prompt)
            chained_context = self.get_chained_context()
            vector_doc = self.get_vectorised_document()
            messages = ChatHistoryUtils.chat_history.messages
            response = chained_context.invoke(
                {
                    "messages": messages,
                    "context": vector_doc
                }
            )
            ChatHistoryUtils.add_message(response, msg_type=enum.ChatMessageType.AI_MESSAGE_TYPE)
            return response
        return str()

    def get_chained_context(self):
        template = ChatPromptTemplate.from_messages(
            [
                (
                    'system', f'{constants.SYSTEM_PROMPT}'
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        chained_context = create_stuff_documents_chain(self.llm, template)
        return chained_context

    def get_knowledge_base(self, file_path=constants.CHAT_KNOWLEDGE_BASE_PATH):
        loader = TextLoader(file_path)
        return loader.load()

    def get_chunked_document(self, data, chunk_size=constants.CHUNK_SIZE, chunk_overlap=constants.CHUNK_OVERLAP):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_documents(data)

    def get_retriever(self, chunked_document, chunk_length=constants.CHUNK_LENGTH):
        vectorstore = Chroma.from_documents(chunked_document, OpenAIEmbeddings())
        retriever = vectorstore.as_retriever(search_kwargs={"k": chunk_length})
        return retriever.invoke(self.prompt)

    def get_vectorised_document(self):
        knowledge_base = self.get_knowledge_base()
        chunked_knowledge_base = self.get_chunked_document(knowledge_base)
        return self.get_retriever(chunked_knowledge_base)


class ChatHistoryUtils(object):
    chat_history = ChatMessageHistory()

    @classmethod
    def add_message(cls, message, msg_type=enum.ChatMessageType.USER_MESSAGE_TYPE):
        if msg_type == enum.ChatMessageType.USER_MESSAGE_TYPE:
            cls.chat_history.add_user_message(message)
        elif msg_type == enum.ChatMessageType.AI_MESSAGE_TYPE:
            cls.chat_history.add_ai_message(message)
        else:
            user_msg_type = enum.ChatMessageType.USER_MESSAGE_TYPE.value
            ai_msg_type = enum.ChatMessageType.AI_MESSAGE_TYPE.value
            unsupported_err_msg = (f"Only '{user_msg_type}' and '{ai_msg_type}' types are supported; "
                                   f"'{msg_type}' is not supported.")
            raise serializers.ValidationError(detail={"message": unsupported_err_msg})
