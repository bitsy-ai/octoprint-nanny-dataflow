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
    async def get_experiment(self, experiment_id=1):
        async with print_nanny_client.ApiClient(self._api_config) as api_client:
            api_instance = print_nanny_client.api.ml_ops_api.MlOpsApi(api_client=api_client)
            experiments = await api_instance.experiments_retrieve(experiment_id)
            return experiments[0]

    @backoff.on_exception(
        backoff.expo,
        aiohttp.ClientConnectionError,
        max_time=MAX_BACKOFF_TIME,
        jitter=backoff.random_jitter,
    )
    async def get_model_artifact(self, model_artifact_id):
        async with print_nanny_client.ApiClient(self._api_config) as api_client:
            api_instance = print_nanny_client.api.ml_ops_api.MlOpsApi(api_client=api_client)
            artifacts  = await api_instance.model_artifacts_retrieve( model_artifact_id)
            return artifacts

