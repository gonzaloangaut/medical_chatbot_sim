from sentence_transformers import SentenceTransformer, util


class SemanticRetriever:
    """
    Class that handles the retriever.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        embedder=None,
        threshold: float = 0.3,
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
        threshold : float
            The minimum cosine similarity threshold.
        chunks : list
            List to store the chunks of the context.
        embeddings : torch.Tensor
            Vector representations of the indexed search keys.
        """
        self.device = "cpu"
        self.model_name = model_name
        self.threshold = threshold

        # Load Embeddings Model
        if embedder is None:
            self.embedder = SentenceTransformer(
                model_name,
                device=self.device,
            )
        else:
            self.embedder = embedder

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
                # If there is not separation, use everything
                self.search_keys.append(block)
                self.chunks.append(block)

        # Vectorize only the symptoms
        self.embeddings = self.embedder.encode(self.search_keys, convert_to_tensor=True)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Search for the most similar pieces of text to the query.

        Parameters
        ----------
        query : str
            The user's query.
        top_k : int | None
            The number of results to return.

        Returns
        ----------
        results : list[dict]
            A list of dictionaries containing the corpus_id, score, and chunk.
        """
        # If top_k is not specified, return all results
        if top_k is None:
            top_k = len(self.chunks)

        # Convert the query to numbers
        query_embedding = self.embedder.encode(
            query,
            convert_to_tensor=True,
        )

        # Search for cosine simililarity and gives the best results
        hits = util.semantic_search(
            query_embedding,
            self.embeddings,
            top_k=top_k,
        )[0]

        # Extract the results
        results = []
        for hit in hits:
            corpus_id = hit["corpus_id"]

            results.append(
                {
                    "corpus_id": corpus_id,
                    "score": float(hit["score"]),
                    "chunk": self.chunks[corpus_id],
                }
            )

        return results

    def retrieve(self, query: str) -> str | None:
        """
        Search the piece of text more similar to the query.

        Parameters
        ----------
        query : str
            The user's query.

        Returns
        ----------
        text : str | None
            The text found or None if no relevant context is found.

        Notes
        -----
        A minimum cosine similarity threshold is applied to avoid
        hallucinated responses when no relevant context is found.
        """
        # Search for the best result
        results = self.search(query, top_k=1)

        # If no results are found or the score is below the threshold, return None
        if not results:
            return None

        # Get the best result
        best_result = results[0]

        # Check if the score is below the threshold
        if best_result["score"] < self.threshold:
            return None

        return best_result["chunk"]
