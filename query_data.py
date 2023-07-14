"""Create a ChatVectorDBChain for question/answering."""
import os
from dotenv import load_dotenv

from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.chat_vector_db.prompts import (
    CONDENSE_QUESTION_PROMPT,
    QA_PROMPT)  # Prompts can be viewed at
                # https://github.com/hwchase17/langchain/blob/master/langchain/chains/chat_vector_db/prompts.py
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.base import VectorStore

MODEL = "gpt-4-0613"
TEMPERATURE = 0.3

# Load .env
load_dotenv()

# OpenAI Key for OpenAI API
openai_api_key = os.environ['OPENAI_API_KEY']

def get_chain(
    vectorstore: VectorStore, question_handler, stream_handler, tracing: bool = False
) -> ConversationalRetrievalChain:
    """Create a ConversationalRetrievalChain for question/answering."""
    # Construct a ConversationalRetrievalChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    question_gen_llm = ChatOpenAI(
        model=MODEL,
        temperature=TEMPERATURE,
        verbose=True,
        callback_manager=question_manager,
    )
    streaming_llm = ChatOpenAI(
        model=MODEL,
        temperature=TEMPERATURE,
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
    )

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
    )
    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=QA_PROMPT, callback_manager=manager
    )

    qa = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
    )
    return qa
