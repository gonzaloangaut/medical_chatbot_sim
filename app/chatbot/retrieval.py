from sentence_transformers import SentenceTransformer, util

from app.chatbot.models import Chunk, RetrievedChunk


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

    def _build_legacy_embedding_text(
        self,
        search_keys: str,
        content: str,
    ) -> str:
        """
        Build the legacy document representation used for embedding.
        """
        if self.representation == "search_keys":
            return search_keys

        if self.representation == "content":
            return content

        if self.representation == "search_keys_and_content":
            return f"{search_keys}\n{content}"

        raise ValueError(f"Unknown document representation: {self.representation}")

    def _build_embedding_text(self, chunk: Chunk) -> str:
        """
        Build the text representation used to embed a chunk.

        Parameters
        ----------
        chunk : Chunk
            The chunk to represent.

        Returns
        -------
        str
            Text representation used by the embedding model.
        """
        return f"{chunk.title}\n\n{chunk.section}\n\n{chunk.content}"

    def ingest_chunks(self, chunks: list[Chunk]) -> None:
        """
        Ingest chunks and compute their embeddings.

        Parameters
        ----------
        chunks : list[Chunk]
            The chunks to ingest.
        """
        # Store the chunks
        self.chunks = chunks

        # Build the embedding texts for each chunk
        embedding_texts = [self._build_embedding_text(chunk) for chunk in chunks]

        # Compute the embeddings for the chunks
        self.embeddings = self.embedder.encode(
            embedding_texts,
            convert_to_tensor=True,
        )

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
                self._build_legacy_embedding_text(
                    search_keys=keys,
                    content=content,
                )
            )

        # Encode the selected document representation
        self.embeddings = self.embedder.encode(
            embedding_texts,
            convert_to_tensor=True,
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Search for the chunks most similar to the query.

        Parameters
        ----------
        query : str
            The user's query.
        top_k : int | None
            The number of results to return.

        Returns
        -------
        list[RetrievedChunk]
            The retrieved chunks and their similarity scores.
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
                RetrievedChunk(
                    chunk=self.chunks[corpus_id],
                    score=float(hit["score"]),
                )
            )

        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant chunks for a query.

        Parameters
        ----------
        query : str
            The user's query.
        top_k : int, optional
            The maximum number of chunks to retrieve.

        Returns
        -------
        list[RetrievedChunk]
            The ranked retrieved chunks. An empty list is returned if no
            relevant context is found.

        Notes
        -----
        The similarity threshold is applied to the highest-ranked result
        as a retrieve-or-reject decision.
        """
        # Search for the most relevant chunks
        results = self.search(query, top_k=top_k)

        # If no results are found or the top result is below the threshold, 
        # return an empty list
        if not results:
            return []

        if results[0].score < self.threshold:
            return []

        # Return the ranked results
        return results
