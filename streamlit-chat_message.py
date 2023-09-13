
import openai
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from PIL import Image

openai.api_key  = st.secrets['OPENAI_API_KEY']

llm_name = "gpt-3.5-turbo-16k-0613"

persist_directory = 'docs/chroma/'

embedding = OpenAIEmbeddings()
vectordb = Chroma(
    embedding_function=embedding,
    persist_directory=persist_directory
)


retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
# create a chatbot chain. Memory is managed externally.

if 'chain' not in st.session_state:
    st.session_state['chain'] = chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(model_name=llm_name, temperature=0, max_tokens=2048), 
    memory=memory,
    retriever=retriever, 
    return_source_documents=True,
    return_generated_question=True,
    condense_question_llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo'))


if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello. Ask me anything about Hopstack!"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey!"]
    
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

#container for the chat history
response_container = st.container()
#container for the user's text input
container = st.container()

def clear_text():
    st.session_state["input"] = " "
    
with container:
    user_input = st.text_input(" ", placeholder="Ask me anything about Hopstack here", key='input', on_change=clear_text)
            
    if user_input:
        output = st.session_state.chain({"question": user_input})
        output = output['answer']
        chat_history=st.session_state["chat_history"]

        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        st.session_state.chat_history.append(chat_history)
        
        
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            
            with st.chat_message("user", avatar='https://raw.githubusercontent.com/megha-hopstack/streamlit-semanticsearch/main/person.png'):
                st.markdown(st.session_state["past"][i])
            with st.chat_message("assistant", avatar='https://raw.githubusercontent.com/megha-hopstack/streamlit-semanticsearch/main/hopstacklogo.png'):
                st.markdown(st.session_state["generated"][i])
            
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

            #MainMenu {visibility: hidden;}