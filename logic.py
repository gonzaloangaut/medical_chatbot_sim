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
        embedder : obj
            The embedder used.
        chunks : list
            List to store the chunks of the context.
        embeddings : torch.Tensor
            Vector representations of the indexed search keys.
        is_indexed : bool
            Boolean to detect if the context is indexed or not.

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
        # If GPU:
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     dtype="auto",
        #     device_map="auto"
        # )
        # Forced Float32 and cpu for Docker
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True
        )
        self.model.eval()

        # Load Embeddings Model
        # self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.embedder = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device=self.device,
        )

        # Variables to save the vectorial database
        self.chunks = []
        self.embeddings = None
        self.is_indexed = False

    def _ingest_context(self, context_text: str):
        """
        Split the text and vectorize it. This is done only once.

        Parameters
        ----------
        context_text : str
            The text to be analized.

        Notes
        -----
        The context text must follow this structure:

        - Blocks separated by '###'
        - Optional separation inside each block using '@@@':
            <search keys> @@@ <official protocol text>

        Only the search keys are embedded, while the full protocol
        text is passed to the LLM.
        """
        # Separate the text
        raw_blocks = context_text.split("###")

        # Vectorize the symptoms
        self.search_keys = []
        # Protocols given to the LLM
        self.chunks = []

        for block in raw_blocks:
            block = block.strip()
            if not block:
                continue

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

        Notes
        -----
        A minimum cosine similarity threshold of 0.3 is applied to avoid
        hallucinated responses when no relevant context is found.
        """
        # Convert the query to numbers
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)

        # Search for cosine simililarity and gives the best result
        hits = util.semantic_search(query_embedding, self.embeddings, top_k=1)

        # Extract the result
        best_hit = hits[0][0]
        score = best_hit["score"]
        doc_id = best_hit["corpus_id"]

        # Security filter
        if score < 0.3:
            return None

        return self.chunks[doc_id]

    def generate_response(self, chat_history: List[Dict[str, str]], context: str):
        """
        Generate a response given the chat history and the medical context.

        This method performs:
        1. Context indexing (if needed)
        2. Semantic retrieval using a RAG approach
        3. Prompt construction based on retrieved context
        4. LLM generative inference

        Parameters
        ----------
        chat_history : List[Dict[str, str]]
            Conversation history containing user and assistant messages.
        context : str
            Medical knowledge base used for retrieval and grounding.

        Returns
        -------
        response : str
            The response generated by the assistant.

        Notes
        -----
        The chat history is expected to follow this structure:
        [{"role": "user", "content": "Hola"},
        {"role": "assistant", "content": "Hola..."}]

        In this model we use a lightweight RAG pipeline over a plain text knowledge
        base. It is intentionally kept simple to allow fast iteration, and can be
        extended later if needed.
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
            context_to_use = (
                "No hay información específica en la base de datos."
                "Respuesta obligatoria: Lo siento, no hay información al respecto."
            )
        # print("CONTEXTO: ", context_to_use)
        full_prompt = f"""
        Instrucciones: Eres un asistente médico.
        Responde DIRECTAMENTE al paciente (usa "usted") basándote SOLO en el TEXTO OFICIAL.
        Usa oraciones completas.
        Sé empático con el paciente.

        TEXTO OFICIAL:
        {context_to_use}

        ---
        PREGUNTA DEL PACIENTE: {last_user_message}
        
        TU RESPUESTA AL PACIENTE:
        """

        # Message structure
        messages = [
            {"role": "system", "content": "Eres un asistente médico útil y preciso."},
            {"role": "user", "content": full_prompt},
        ]

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        return response.strip()
