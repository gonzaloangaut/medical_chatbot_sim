"""
Module containing the MedicalAssistance class.

Classes:
    - MedicalAssistance: Class that represents a chatbot for medical assistance.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util

class MedicalAssistance:
    """
    Class that represents a chatbot for medical assistance.

    This class handles the chatbot to generate responses.
    """

    def __init__(self):
        """
        Initialize a new bot.

        Attributes
        ----------
        device : str
            The device that is being used to run the program.
        tokenizer : obj
            The tokenizer used by the bot.
        model : obj
            The model of LLM (or SLM) used.
        
        Notes
        -----
        In this project we will use Qwen, which is a family of LLMs. Because of the 
        execution time, we will load one model with few parameters.
        """
        # Use GPU if available
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use CPU for Docker
        self.device = "cpu"
        # torch.set_num_threads(4)
        
        # Defined the model to use
        # In this case, we use Qwen (SLM)
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"

        # Load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     dtype="auto",
        #     device_map="auto"
        # )
        # Forced Float32 and cpu for Docker
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        self.model.eval()

        # Load Embeddings Model
        # self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=self.device)
        
        # Variables to save the vectorial database
        self.chunks = []
        self.embeddings = None
        self.is_indexed = False

    def _ingest_context(self, context_text: str):
        """
        Split the text and vectorize it. This is done only once.
        
        Parameters
        ----------
        context_tex : str
            The text to be analized.
        """
        # Split the text by ###
        raw_chunks = context_text.split("###")
        
        # Clean empty spaces
        self.chunks = [c.strip() for c in raw_chunks if c.strip()]
        
        # Embeddings
        self.embeddings = self.embedder.encode(self.chunks, convert_to_tensor=True)
        # Say that it is already indexed
        self.is_indexed = True
        try:
            # Separate the text
            raw_blocks = context_text.split("###")

            # Vectorize the symptoms            
            self.search_keys = []
            # Protocols given to the LLM
            self.chunks = []

            for block in raw_blocks:
                block = block.strip()
                if not block: continue

                # Search for the separation by @@@
                if "@@@" in block:
                    # We look for the separation and add the chunk
                    keys, content = block.split("@@@", 1)
                    self.search_keys.append(keys.strip())
                    self.chunks.append(content.strip())
                else:
                    # Id there is not separation, use everything
                    self.search_keys.append(block)
                    self.chunks.append(block)

            # Vectorize only the symptoms
            self.embeddings = self.embedder.encode(self.search_keys, convert_to_tensor=True)
            self.is_indexed = True
            
        except Exception as e:
            self.chunks = ["Error cargando base de datos."]
            self.embeddings = self.embedder.encode(["Error"], convert_to_tensor=True)
            self.is_indexed = True
    def _retrieve(self, query: str):
        """
        Search the piece of text more similar to the query.
        
        Parameters
        ----------
        query : str
            The user's query.

        Returns
        ----------
        text : str
            The text found.
        """
        # Convert the query to numbers
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        
        # Search for cosine simililarity and gives the best result
        hits = util.semantic_search(query_embedding, self.embeddings, top_k=1)
        
        # Extract the result
        best_hit = hits[0][0]
        score = best_hit['score']
        doc_id = best_hit['corpus_id']
        
        # Security filter
        if score < 0.3:
            return None
            
        return self.chunks[doc_id]

    def generate_response(self, chat_history: List[Dict[str, str]], context: str):
        """
        Generate a response given the chat history and context.

        The history has the following structure:
        [{"role": "user", "content": "Hola"}, 
        {"role": "assistant", "content": "Hola..."}]
        
        Parameters
        ----------
        chat_history : List[Dict[str, str]]
            The chat's history. It contains every user's query and the given answer.
        context : str
            The medical context to be used.

        Returns
        ----------
        response : str
            The response generated by the bot.

        Notes
        -----
        In this simple model we are only using context by RAG and a simple .txt.
        In real-world problems, we should use a big quantity of data, and because
        of that, embeddings would be important to manage this data.
        """
        # Check if the context is indexed or we index it now
        if not self.is_indexed:
            self._ingest_context(context)

        # Get the last query
        last_user_message = chat_history[-1]["content"]

        # RAG: Search only the relevant information
        relevant_info = self._retrieve(last_user_message)

        if relevant_info:
            context_to_use = relevant_info
        else:
            context_to_use = "No hay información específica en la base de datos." \
            "Respuesta obligatoria: Lo siento, no hay información al respecto."

        full_prompt = f"""
        Instrucciones: Eres un asistente médico.
        Responde DIRECTAMENTE al paciente (usa "usted") basándote SOLO en el TEXTO OFICIAL.
        Usa oraciones completas.

        TEXTO OFICIAL:
        {context_to_use}

        ---
        PREGUNTA DEL PACIENTE: {last_user_message}
        
        TU RESPUESTA AL PACIENTE:
        """

        # Message structure
        messages = [
            {"role": "system", "content": "Eres un asistente médico útil y preciso."},
            {"role": "user", "content": full_prompt}
        ]        

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        # Generate the output        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )

        # Decode the new response
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True)

        return response.strip()