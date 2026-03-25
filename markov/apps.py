import logging
from django.apps import AppConfig

logger = logging.getLogger(__name__)

class MarkovConfig(AppConfig):
    name = 'markov'
    
    def ready(self):
        """Initialise le wrapper au démarrage de Django (en background)."""
        import threading
        def init_wrapper():
            try:
                logger.info("⏳ Starting MarkovAnalyzer initialization in background thread...")
                from .markov_analyzer_wrapper import get_analyzer
                analyzer = get_analyzer()
                logger.info(f"✅ MarkovAnalyzer ready with {len(analyzer.experiments)} experiments")
            except Exception as e:
                logger.error(f"❌ Error initializing wrapper: {e}")
        
        # Démarrer dans un thread séparé pour ne pas bloquer le serveur
        # Ce thread continuera à s'exécuter en arrière-plan
        thread = threading.Thread(target=init_wrapper, daemon=True, name="MarkovAnalyzerLoader")
        thread.start()
        logger.info("✅ MarkovAnalyzer loader thread started (initialization will happen in background)")

