"""STEP 1: Read the raw document from the data folder."""

from langchain_community.document_loaders import TextLoader
from hr_assistant import config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def load_document(file_path: str = config.DATA_FILE_PATH):

    """Load a .txt file and return it as a list of Langchain Document objects"""

    logger.info("LOADING DOCUMENTS from document loader: %s", file_path)

    loader = TextLoader(file_path,encoding="utf-8")

    documents = loader.load()

    logger.info("Loaded %d docuemnt(s)",len(documents))

    return documents


