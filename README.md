# Medical Query Assistant

This project is a Medical Query Assistant application that leverages machine learning models to process and respond to user queries related to health and medical topics. It uses various tools and libraries to provide text and audio responses.

## Features

- **Text and Audio Input**: Users can input queries via text or audio.
- **Multiple Response Sources**: Provides responses from Direct Model, ChromaDB, and Retrieval-Augmented Generation (RAG).
- **Text-to-Speech**: Converts text responses to audio using Google Text-to-Speech (gTTS).
- **User Interface**: Built with Gradio for easy interaction.

## Installation

1. **Clone the Repository**:
bash
git clone https://github.com/K7alid/Graduation_Project_of_DEPI.git
cd Graduation_Project_of_DEPI


2. **Install Required Libraries**:
   Ensure you have Python installed, then run:
bash
pip install --upgrade huggingface_hub
pip install chromadb transformers torch langchain tokenizers datasets
pip install chroma-migrate
pip install --upgrade bitsandbytes
pip install gradio
pip install gTTS


3. **Mount Google Drive** (if using Colab):
python
from google.colab import drive
drive.mount('/content/drive')


## Usage

1. **Prepare Text Data**: Provide the path to your text file or set it to `None` to use the existing database.
python
text_file_path = "/content/cleaned_docs_output_fixed1.txt"  # Replace with your text file path or set to None


2. **Run the Application**:
   Initialize and launch the assistant:
python
assistant = MedicalQueryAssistant()
assistant.launch_interface()


3. **Interact with the Interface**:
   - Enter your medical query as text or speak into the microphone.
   - Choose to enable text-to-speech for any response type.

## Components

- **EmbeddingHandler**: Generates text embeddings.
- **TextProcessor**: Loads and splits text into chunks.
- **ChromaDBHandler**: Manages a database of text embeddings.
- **InferenceModel**: Generates responses using a language model.
- **TextToSpeechHandler**: Converts text to speech.
- **TranscriptionHandler**: Transcribes audio input.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing pre-trained models.
- [Gradio](https://gradio.app/) for the user interface framework.
- [Google Text-to-Speech](https://pypi.org/project/gTTS/) for TTS functionality.
