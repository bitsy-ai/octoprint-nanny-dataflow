import aiohttp
import backoff
import urllib
import print_nanny_client
MAX_BACKOFF_TIME = 120

class RestAPIClient:
    """
    webapp rest API calls and retry behavior
    """

    def __init__(self, api_token: str, api_url: str):
        self.api_url = api_url
        self.api_token = api_token

    @property
    def _api_config(self):
        parsed_uri = urllib.parse.urlparse(self.api_url)
        host = f"{parsed_uri.scheme}://{parsed_uri.netloc}"
        config = print_nanny_client.Configuration(host=host)

        config.access_token = self.api_token
        return config
    
    @backoff.on_exception(
        backoff.expo,
        aiohttp.ClientConnectionError,
        max_time=MAX_BACKOFF_TIME,
        jitter=backoff.random_jitter,
    )
    async def get_active_experiment(self):
        async with print_nanny_client.ApiClient(self._api_config) as api_client:
            request = print_nanny_client.ExperimentRequest(active=True)
            api_instance = print_nanny_client.api.ml_ops_api.MlOpsApi(api_client=api_client)
            experiments = await api_instance.experiments_retrieve(
                request
            )
            return experiments[0]