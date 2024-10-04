from google.colab import drive
drive.mount('/content/drive')


# %% Install Required Libraries
!pip install --upgrade huggingface_hub
!pip install chromadb transformers torch langchain tokenizers datasets
!pip install chroma-migrate
!pip install --upgrade bitsandbytes
!pip install gradio
!pip install gTTS

# %% Import Libraries
import gradio as gr
import re
from uuid import uuid4
from transformers import (
    AutoTokenizer,
    AutoModel,
    pipeline,
)
import torch
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from huggingface_hub import InferenceClient
from gtts import gTTS
import io

    # %% Define Helper Classes

class EmbeddingHandler:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L12-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        print(f"Loaded embedding model '{model_name}' on {self.device}")

    def max_token_length(self, txt_list):
        max_length = max(len(re.findall(r'\w+', txt)) for txt in txt_list)
        return f"Max Token Length: {max_length} tokens"

    def embed_text(self, texts, batch_size=32):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

class TextProcessor:
    def __init__(self, text_path=None):
        self.text_path = text_path
        self.text_raw = self.load_text()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " "],
        )
        self.text_chunks = self.split_text()

    def load_text(self):
        if self.text_path:
            try:
                with open(self.text_path, "r", encoding="utf-8") as f:
                    text = f.read()
                print(f"Loaded text from '{self.text_path}'")
                return text
            except Exception as e:
                print(f"Error loading text file: {e}")
                return ""
        else:
            print("No text file path provided. Skipping text loading.")
            return ""

    def split_text(self):
        if self.text_raw:
            chunks = self.text_splitter.split_text(self.text_raw)
            print(f"Total number of chunks: {len(chunks)}")
            return chunks
        else:
            return []

class ChromaDBHandler:
    def __init__(self, persist_directory='/content/drive/MyDrive/db_embeddings/chromadb_data', collection_name="medical_embeddings"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        print(f"Connected to ChromaDB at '{persist_directory}', collection '{collection_name}'")

    def add_documents(self, embeddings, documents):
        ids = [str(uuid4()) for _ in range(len(documents))]
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            ids=ids,
        )
        print(f"Added {len(documents)} documents to ChromaDB")

    def query(self, query_embedding, n_results=5):
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
        )
        return results

class InferenceModel:
    def __init__(self, model_id='mistralai/Mixtral-8x7B-Instruct-v0.1', token=None):
        self.client = InferenceClient(
            model_id,
            token=token,
        )
        print(f"Initialized InferenceClient with model '{model_id}'")

    def generate_response(self, prompt, max_new_tokens=150):
        response = self.client.text_generation(prompt, max_new_tokens=max_new_tokens)
        if isinstance(response, str):
            return response
        else:
            return "Error: Unexpected response format."

class TextToSpeechHandler:
    def __init__(self):
        pass

    def text_to_speech_gtts(self, text):
        try:
            tts = gTTS(text)
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            return audio_bytes.read()
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

class TranscriptionHandler:
    def __init__(self, model_name="openai/whisper-base.en"):
        self.transcriber = pipeline("automatic-speech-recognition", model=model_name)
        print(f"Initialized transcription model '{model_name}'")

    def transcribe(self, audio):
        try:
            sr, y = audio

            # Convert to mono if stereo
            if y.ndim > 1:
                y = y.mean(axis=1)

            y = y.astype(np.float32)
            y /= np.max(np.abs(y))

            # Transcribe the audio
            result = self.transcriber({"sampling_rate": sr, "raw": y})
            return result["text"]
        except Exception as e:
            return f"An error occurred during transcription: {str(e)}"

# %% Define the Main Application Class
history_messages=[]
class MedicalQueryAssistant:
    def __init__(self, text_path=None, chroma_persist_dir='/content/drive/MyDrive/db_embeddings/chromadb_data'):
        # Initialize components
        self.embedding_handler = EmbeddingHandler()
        self.text_processor = TextProcessor(text_path=text_path)
        self.chroma_handler = ChromaDBHandler(persist_directory=chroma_persist_dir)
        self.tts_handler = TextToSpeechHandler()
        self.transcription_handler = TranscriptionHandler()
        self.client = InferenceClient(
        'mistralai/Mixtral-8x7B-Instruct-v0.1',
        token="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",
            )

        # Initialize Inference Model with your Hugging Face token
        # hf_token = "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"  # Replace with your token or handle securely
        # self.inference_model = InferenceModel(token=hf_token)

        # If text is provided, process and store embeddings
        if self.text_processor.text_chunks:
            embeddings = self.embedding_handler.embed_text(self.text_processor.text_chunks, batch_size=32)
            print(f"Generated {len(embeddings)} embeddings")
            self.chroma_handler.add_documents(embeddings, self.text_processor.text_chunks)
        else:
            print("No text chunks to process. Ensure the database has been populated.")

    def query_medical(self, query, n_results=5):
        query_embedding = self.embedding_handler.embed_text([query])[0]
        results = self.chroma_handler.query(query_embedding, n_results=n_results)
        return results

    def direct_model_query(self, max_new_tokens=150, history = []):
        sys_message = (
            f"You are Medico,So your name is Medico and if anyone asked you for your name you can tell him that your name is Medico and you are an AI Medical Assistant with expertise in a wide range of health topics. "
            f"Your goal is to provide clear, accurate, and helpful information within the limit of {max_new_tokens} tokens. "
            f"Always prioritize user safety and privacy. If you are unsure about a medical question, recommend consulting a healthcare professional. "
            f"Advise users to consult their doctor for medication alternatives or replacements if their prescribed medication is unavailable. "
            f"Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. "
            f"Use three sentences maximum and keep the answer concise. "
            f"Please note that you are a medical chatbot and should only answer questions related to health and medical topics."
        )

        messages = [
        {"role": "system", "content": sys_message},
        ]+ history






        # prompt = f"{sys_message}\n\nUser: {user_query}\n\nAssistant:"
        # response = self.inference_model.generate_response(prompt, max_new_tokens=max_new_tokens)

        response = ''
        for message in self.client.chat_completion(
                  messages=messages,
                  stream=True,
                  max_tokens=500,
                  temperature=0.7
              ):
                  response += message.choices[0].delta.content

        # Use the InferenceClient to generate a response
        # response = client.text_generation(prompt, max_new_tokens=max_new_tokens)

        # Check if the response is a string and handle accordingly
        if isinstance(response, str):
            answer = response
        else:
            answer = "Error: Unexpected response format."

        return answer

    def rag_medical_hf(self, user_query ,max_new_tokens=150, history = []):


        print(user_query)

        # Retrieve relevant medical text from ChromaDB
        retrieved_results = self.query_medical(user_query)
        print(retrieved_results)

        # Concatenate retrieved documents
        retrieved_texts = "; ".join(retrieved_results["documents"][0])

        # Construct the system prompt
        system_role_definition = (
            f"You are Medico,So your name is Medico and if anyone asked you for your name you can tell him that your name is Medico and you are an AI Medical Assistant with expertise in a wide range of health topics. "
            f"Your goal is to provide clear, accurate, and helpful information within the limit of {max_new_tokens} tokens. "
            f"Always prioritize user safety and privacy. If you are unsure about a medical question, recommend consulting a healthcare professional. "
            f"Advise users to consult their doctor for medication alternatives or replacements if their prescribed medication is unavailable. "
            f"Use the following pieces of retrieved context {retrieved_results} to answer the question . If you don't know the answer, say that you don't know. "
            f"Use three sentences maximum and keep the answer concise. "
            f"Please note that you are a medical chatbot and should only answer questions related to health and medical topics."
        )
        # user_query_complete = f"What are the medical insights related to: {user_query}?"
        # prompt = f"{system_role_definition}\nUser Query: {user_query_complete}\nInformation: {retrieved_texts}\nAnswer:"

        # response = self.inference_model.generate_response(prompt, max_new_tokens=max_new_tokens)


        messages = [
          {"role": "system", "content": system_role_definition},
        ]+ history

        response = ''
        for message in self.client.chat_completion(
                messages=messages,
                stream=True,
                max_tokens=500,
                temperature=0.7
            ):
                response += message.choices[0].delta.content


        # Use the InferenceClient to generate a response
        # response = client.text_generation(prompt, max_new_tokens=max_new_tokens)

        # Print the response to understand its structure
        # print("Response:", response)

        # Check if the response is a string and handle accordingly
        if isinstance(response, str):
            content = response
        else:
            content = "Error: Unexpected response format."

        return content


    def get_responses(self, text_input, audio_input, tts_direct, tts_chroma, tts_rag):
        # Determine user query
        if audio_input is not None:
            user_query = self.transcription_handler.transcribe(audio_input)
        elif text_input is not None and text_input.strip() != "":
            user_query = text_input
        else:
            return "No input provided", None, "No input provided", None, "No input provided", None

        history_messages.append({"role": "user", "content": user_query})
        # Generate responses
        direct_response = self.direct_model_query(max_new_tokens=300 , history=history_messages)
        chromadb_results = self.query_medical(user_query)
        chromadb_response = "; ".join(chromadb_results["documents"][0])
        rag_response = self.rag_medical_hf(user_query= user_query, max_new_tokens=300 , history=history_messages)
        history_messages.append({"role": "assistant", "content": rag_response})

        # Convert to speech if required
        direct_audio = self.tts_handler.text_to_speech_gtts(direct_response) if tts_direct else None
        chroma_audio = self.tts_handler.text_to_speech_gtts(chromadb_response) if tts_chroma else None
        rag_audio = self.tts_handler.text_to_speech_gtts(rag_response) if tts_rag else None

        return (
            direct_response, direct_audio,
            chromadb_response, chroma_audio,
            rag_response, rag_audio
        )

    def launch_interface(self):
        medical_demo = gr.Interface(
            fn=self.get_responses,
            inputs=[
                gr.Textbox(label="User Query", placeholder="Type your query here..."),
                gr.Audio(sources="microphone", type="numpy", label="Or speak your query"),
                gr.Checkbox(label="Enable TTS for Direct Model Response", value=False),
                gr.Checkbox(label="Enable TTS for ChromaDB Response", value=False),
                gr.Checkbox(label="Enable TTS for RAG Response", value=False)
            ],
            outputs=[
                gr.Textbox(label="Direct Model Response Text"),
                gr.Audio(label="Direct Model Response Audio", format="mp3"),
                gr.Textbox(label="ChromaDB Response Text"),
                gr.Audio(label="ChromaDB Response Audio", format="mp3"),
                gr.Textbox(label="RAG Response Text"),
                gr.Audio(label="RAG Response Audio", format="mp3")
            ],
            title="Medical Query Assistant",
            description="Enter your medical query as text or speak into the microphone to get responses from Direct Model, ChromaDB, and RAG. Enable text-to-speech for any response."
        )
        medical_demo.launch(debug=True)

# %% Initialize and Launch the Assistant

# Provide the path to your text file or set to None to use the existing database
text_file_path = "/content/cleaned_docs_output_fixed1.txt"  # Replace with your text file path or set to None

assistant = MedicalQueryAssistant()
assistant.launch_interface()


