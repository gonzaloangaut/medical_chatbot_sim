from sentence_transformers import SentenceTransformer, util

class SemanticRetriever:
    """
    Class that handles the retriever.
    """
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        """
        Initialize a new retriever.

        Attributes
        ----------
        device : str
            The device that is being used to run the program.
        model_name : str
            The name of the model used for transformers.
        embedder : obj
            The embedder used.
        chunks : list
            List to store the chunks of the context.
        embeddings : torch.Tensor
            Vector representations of the indexed search keys.
        is_indexed : bool
            Boolean to detect if the context is indexed or not.
        """
        self.device = "cpu"
        self.model_name = model_name

        # Load Embeddings Model
        # self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.embedder = SentenceTransformer(
            model_name,
            device=self.device,
        )

        # Variables to save the vectorial database
        self.search_keys = []
        self.chunks = []
        self.embeddings = None

    def ingest_context(self, context_text: str):
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

    def retrieve(self, query: str):
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