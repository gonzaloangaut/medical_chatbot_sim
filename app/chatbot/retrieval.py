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
            Vector representations of the indexed document representations.
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

        # Indexed retrieval data
        self.chunks = []
        self.embeddings = None

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
