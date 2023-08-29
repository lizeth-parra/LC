import os

from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import streamlit as st

os.environ['OPENAI_API_KEY'] = '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'
default_doc_name = 'doc.pdf'

#Método que tiene parámetros y establece que si no se tiene cargado ningún documento
#se redirige al pdf de internet respondiendo la pregunta "question"
def process_doc(
        path: str = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf',
        is_local: bool = False,
        question: str = 'Cuáles son los autores del pdf?'
):
    #Metodo de carga de archivo en caso no queramos usar el pdf designado por defecto <PyPDFLoader(path)>
    _, loader = os.system(f'curl -o {default_doc_name} {path}'), PyPDFLoader(f"./{default_doc_name}") if not is_local \
        else PyPDFLoader(path)

    #Carga de documento y particinamiento para analisis del pdf por partes (metodo split)
    doc = loader.load_and_split()

    print(doc[-1])
    # se usa para crear el índice de la tienda de vectores usando el fragmento
    # del documento después de la división del texto y la OpenAIEmbeddings()función
    # como argumentos de entrada.
    db = Chroma.from_documents(doc, embedding=OpenAIEmbeddings())
    # RetrievalQA-cadena de pregunta-respuesta que toma como argumentos de entrada el LLM
    # a través del llmparámetro, el tipo de cadena a utilizar a través del
    # chain_typeparámetro y el recuperador a través del retrieverparámetro.
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type='stuff', retriever=db.as_retriever())

    st.write(qa.run(question))
    # print(qa.run(question))


def client():
    st.title('Manage LLM with LangChain')
    # agregue widgets de entrada que permitan a los
    # usuarios cargar archivos de texto usando st.file_uploader()
    uploader = st.file_uploader('Upload PDF', type='pdf')

    if uploader:
        with open(f'./{default_doc_name}', 'wb') as f:
            f.write(uploader.getbuffer())
        st.success('PDF saved!!')

    question = st.text_input('Generar un resumen de 20 palabras sobre el pdf',
                             placeholder='Give response about your PDF', disabled=not uploader)

    if st.button('Send Question'):
        if uploader:
            process_doc(
                path=default_doc_name,
                is_local=True,
                question=question
            )
        else:
            st.info('Loading default PDF')
            process_doc()


if __name__ == '__main__':
    client()
    # process_doc()
