
import streamlit as st
import pickle
#from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
import openai
#from openai import error
#from openai import OpenAI
from langchain.llms import OpenAI 
from langchain.chains.question_answering import load_qa_chain

#openai.api_key = os.environ['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']
os.environ["OPENAI_API_KEY"] = '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'


#OPENAI_API_KEY = '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'
with st.sidebar:
    st.title('Chat with PDF :')
    st.markdown('''
    ## About
                
    This application is a large language model-powered chatbot developed using Streamlit, Langchain, and OpenAI  
                ''')



    add_vertical_space(5)
    st.write('Built by [Abdul Rahman](https://www.linkedin.com/in/abdul-rahman-586664226/)')

def main():
    st.header('Chat With PDF')

    #load_dotenv()
    # upload a pdf file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    
    # read pdf file
    if pdf is not None:
        # read pdf file
        pdf_reader = PdfReader(pdf)
      #  st.write(pdf_reader)

        # Extracting text from pdf
        text = ""
        for page in pdf_reader.pages:
            text = text + page.extract_text()

        #st.write(text)

        # Splitting data into chunks
        text_splitter = RecursiveCharacterTextSplitter( chunk_size=1000,
            chunk_overlap= 200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
        
        # COMPUTING THE CROSSPONDING EMBEDDINGS OF EACH CHUNK
        # embeddings = OpenAIEmbeddings()
        # VectorStore= FAISS.from_texts(chunks, embedding=embeddings)
        
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):

            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)

            #st.write('Embedding from divice')

        else:
           
            embeddings = OpenAIEmbeddings()
            VectorStore= FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
            
            #st.write('New Embedding created')
                
        # query
        query= st.text_input('Ask a Question')
        #st.write(query)
          
        if query:
            docs = VectorStore.similarity_search(query=query,k=3)
            
            llm= OpenAI(temperature=0,model_name= 'gpt-3.5-turbo')

            chain = load_qa_chain(llm=llm, chain_type= 'stuff')

            response = chain.run(input_documents= docs, question=query)
            st.write(response)


        






        

        #st.write(chunks)


    


if __name__== '__main__':
    main()

