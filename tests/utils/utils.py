import json
import responses
from typing import Union, Optional


def mock_api_response(
    endpoint: str,
    response_body: Optional[Union[dict, list[dict]]] = None,
    request_type: str = responses.POST,
    status: int = 200,
    stream: bool = False,
):
    """Mock the API responses for the given endpoint"""
    endpoint = f"http://test-servers/api{endpoint}"
    if stream:
        body = "\n".join(json.dumps(item) for item in response_body).encode()
        responses.add(request_type, endpoint, body=body, status=status, stream=True)
    else:
        responses.add(request_type, endpoint, json=response_body, status=status)
