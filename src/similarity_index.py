# import necessary libraries
import tensorflow_hub as hub

# Global variable to cache the loaded model
_model = None


def get_embedding_model():
    """
    Load and return the universal sentence encoder model.
    The model is cached after first load to avoid reloading.
    """
    global _model
    if _model is None:
        _model = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/tensorFlow2/universal-sentence-encoder/2?tfhub-redirect=true")
    return _model


def generate_embeddings(sentences):
    """
    Generate embeddings for a list of sentences using the universal sentence encoder.
    """
    # Load the model (cached after first call)
    embed = get_embedding_model()
    
    # Generate embeddings
    embeddings = embed(sentences)
    
    return embeddings