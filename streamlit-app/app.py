import streamlit as st
from dotenv import load_dotenv #for api calls
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter #spliting texts
from langchain.embeddings.openai import OpenAIEmbeddings #embedding after splitting
from langchain.vectorstores import FAISS #storing into a store so that we can reduce the number of lookups
import pickle
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


#Sidebar contents
with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    
    ''')
    add_vertical_space(5)
    st.write('Made by yash Sinha')
    

def main():
    st.header("chat with PDFGPT")

    load_dotenv() 
    mode_name ="gpt-3.5-turbo"
   

    #upload a PDF file
    pdf = st.file_uploader("upload your PDF", type = "pdf")

    #st.write(pdf)

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        #st.write(pdf_reader)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

       # st.write(text)

        text_splitter = RecursiveCharacterTextSplitter (
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
            )
        chunks = text_splitter.split_text(text=text)

        #embeddings created an object
       # embeddings = OpenAIEmbeddings()
        #VectorStore = FAISS.from_texts(chunks,embedding=embeddings)
        #this whole process of creating is a very expensive process
        #so we shall try to find ways to get around
        #so what we are doing is steroing the file name and seeing 
        #if we have used the same file before then we will not again 
        #do embedding just go to vector store for searching.
         
    #embedding
    store_name = pdf.name[:-4] 
      
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl","rb") as f:
            VectorStore = pickle.load(f)
            st.write("embedding loaded from the disk")
    else:
         embeddings = OpenAIEmbeddings()
         VectorStore = FAISS.from_texts(chunks,embedding=embeddings)
         with open(f"{store_name}.pkl","wb") as f:
            pickle.dump(VectorStore,f)
            st.write("writing on the disk")

#basically pickle is used for writing the files into storage
       # st.write(chunks)


       #Accept user input ie. questions
    query = st.text_input("Ask questions about your PDF file:")
    #st.write(query)

    if query:
        docs = VectorStore.similarity_search(query=query, k= 5)

        llm = OpenAI(model_name ='gpt-3.5-turbo') 
        chain = load_qa_chain(llm= llm, chain_type="stuff")
        response = chain.run(input_documents= docs, question = query)
        st.write(response)





        # st.write(docs)





if __name__ =='__main__':
   main()    






