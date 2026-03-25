import logging
from django.apps import AppConfig

logger = logging.getLogger(__name__)

class MarkovConfig(AppConfig):
    name = 'markov'
    
    def ready(self):
        """Django app ready — pas d'initialisation lourde au démarrage."""
        pass

