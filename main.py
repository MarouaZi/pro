import streamlit as st
import google.generativeai as gen_ai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

#import langchain_google_genai.exceptions



# Configure Streamlit page settings
st.set_page_config(
    page_title="BuddyBOT", #add page title
    page_icon=":brain:",  # add emoji
    layout="centered",  # Page layout option
)
chat_history = []
# Display the chatbot's title on the page
st.title("🤖 ChatBot")
chat_history = []
with st.chat_message("assistant"):# Adding a chat message block for the assistant input.
    st.markdown("Hello, I am your assistant. How can I help you?") # Displaying the assistent first wellcom line.

# Set up Google Gemini-Pro AI model
api_key=gen_ai.configure(api_key="AIzaSyCiPt8B5VpJnwb9ChD6abJ67hjnCu6gvCI")
model = gen_ai.GenerativeModel('gemini-pro')

# Function to translate roles from Gemini-Pro to Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

# Initialize chat session in Streamlit if not already present

# if the chat history is vide creat a new chat

if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])
    
for message in st.session_state.chat_session.history:
    with st.chat_message(translate_role_for_streamlit(message.role)):
        st.markdown(message.parts[0].text)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyCiPt8B5VpJnwb9ChD6abJ67hjnCu6gvCI")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context  just say, "naaaaam", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3, google_api_key="AIzaSyCiPt8B5VpJnwb9ChD6abJ67hjnCu6gvCI")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyCiPt8B5VpJnwb9ChD6abJ67hjnCu6gvCI")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    try:
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        if response is not None and response["output_text"] != "naaaaam":
            return response["output_text"]
        else:
            return False
    except Exception as e:
        print(f"An exception occurred: {e}")
        return False


pdf_docs=['Banque_FR.pdf','banque_AR.pdf']
raw_text = get_pdf_text(pdf_docs)
text_chunks = get_text_chunks(raw_text)
get_vector_store(text_chunks)


# Input field for user's message
user_prompt = st.chat_input("Type your message here...")
if user_prompt:
     # Add user's message to chat session
    #st.session_state.chat_session.append({"role": "user", "text": user_prompt})
    # Add user's message to chat and display it
    st.chat_message("user").markdown(user_prompt)
    # Send user's message to Gemini-Pro and get the response
    docs= user_input(user_prompt)
    # Add bot's response to chat session
    #st.session_state.chat_session.append({"role": "model", "text": docs})
    if docs is not False:
        st.session_state.chat_session.append({"role": "assistant", "context": bot_response})
     # Display bot's response
        with st.chat_message("assistant"):
          st.markdown(docs)
    
    else:
            # Send user's message to Gemini-Pro and get the response
     gemini_response = st.session_state.chat_session.send_message(user_prompt)

            # Display Gemini-Pro's response
     with st.chat_message("assistant"):
                response = gemini_response.text
                st.markdown(response)
                # Adding the AI response into the chat history list
                chat_history.append({"role": "assistant", "content": response})
                # Update Chat History in Sidebar
                st.sidebar.markdown(f"**Assistant:** {response}")   
                pass
     
        