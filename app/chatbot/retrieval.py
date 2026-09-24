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
        representation: str = "search_keys",
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
        representation : str
            The text used for embedding.
        chunks : list
            List to store the chunks of the context.
        embeddings : torch.Tensor
            Vector representations of the indexed document representations.
        """
        self.device = "cpu"
        self.model_name = model_name
        self.threshold = threshold
        self.representation = representation

        # Load Embeddings Model
        if embedder is None:
            self.embedder = SentenceTransformer(
                model_name,
                device=self.device,
            )
        else:
            self.embedder = embedder

        # Indexed retrieval data
        self.search_keys = []
        self.chunks = []
        self.embeddings = None

    def _build_embedding_text(
        self,
        search_keys: str,
        content: str,
    ) -> str:
        """
        Build the document representation used for embedding.
        """
        if self.representation == "search_keys":
            return search_keys

        if self.representation == "content":
            return content

        if self.representation == "search_keys_and_content":
            return f"{search_keys}\n{content}"

        raise ValueError(f"Unknown document representation: {self.representation}")

    def ingest_context(self, context_text: str):
        """
        Parse the knowledge base and compute document embeddings.

        Parameters
        ----------
        context_text : str
            Raw knowledge-base text.

        Notes
        -----
        Blocks must be separated by '###'.

        Each block may optionally follow:

            <search keys> @@@ <content>

        The text used to compute each document embedding depends on
        `self.representation`, while `content` is stored as the chunk
        returned by the retriever.
        """
        # Separate the text
        raw_blocks = context_text.split("###")

        # Store parsed search keys
        self.search_keys = []
        # Protocols given to the LLM
        self.chunks = []
        # Embedding texts
        embedding_texts = []

        for block in raw_blocks:
            block = block.strip()
            if not block:
                continue

            # Search for the separation by @@@
            if "@@@" in block:
                # We look for the separation and add the chunk
                keys, content = block.split("@@@", 1)
                keys = keys.strip()
                content = content.strip()
            else:
                # If there is not separation, use everything
                keys = block
                content = block
            # Append keys and content
            self.search_keys.append(keys)
            self.chunks.append(content)
            # Embedding text used
            embedding_texts.append(
                self._build_embedding_text(
                    search_keys=keys,
                    content=content,
                )
            )

        # Encode the selected document representation
        self.embeddings = self.embedder.encode(
            embedding_texts,
            convert_to_tensor=True,
        )

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
        A minimum similarity threshold is used to reject queries when
        the Top-1 document does not reach the required retrieval score.
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
