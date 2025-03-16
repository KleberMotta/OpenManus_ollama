from app.tool.search.base import WebSearchEngine
import googlesearch

class GoogleSearchEngine(WebSearchEngine):
    
    def perform_search(self, query, num_results = 10, *args, **kwargs):
        """Google search engine."""
        return googlesearch.search(query, num_results=num_results)
