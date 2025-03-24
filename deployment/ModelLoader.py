from sentence_transformers import SentenceTransformer
from BertRelevanceClassifier import BertRelevanceClassifier
import torch
import logging

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DROPOUT_RATE = 0.1
NUM_LABELS = 3
EMBEDDING_SIZE = 768
MODEL_PATH = './chosen_models/model_167'

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ModelLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.embedder, cls._instance.model = cls._load_embed_and_model()
        return cls._instance

    @staticmethod
    def _load_embed_and_model():
        logger.info("Loading emrecan's BERT sentence embedder")
        embedder = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr', device=DEVICE)
        logger.info("Loading predicting model")
        model = BertRelevanceClassifier(NUM_LABELS, EMBEDDING_SIZE, DROPOUT_RATE)
        model.load_state_dict(torch.load(MODEL_PATH)) 
        logger.info("Embedder and model loaded and ready for predicting.")

        return embedder, model