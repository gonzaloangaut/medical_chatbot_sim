from pathlib import Path

def load_knowledge() -> str:
    """
    Load the knowledge base from a text file.

    Returns
    -------
    knowledge : str
        The content of the knowledge base.
    """
    # Define the path to the knowledge file
    project_root = Path(__file__).resolve().parent.parent.parent

    # Define the path to the context file
    context_path = project_root / "data" / "context.txt"

    # Read and return the content of the context file
    return context_path.read_text(encoding="utf-8")