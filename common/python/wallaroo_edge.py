"""
Basic routines for integrating Wallaroo w/ an edge inference endpoint
"""

import logging
import json
import pyarrow as pa
import numpy as np
from httpx import AsyncClient, TimeoutException, HTTPStatusError, NetworkError
from tenacity import AsyncRetrying, stop_after_attempt, retry_if_exception_type
from typing import (
    TYPE_CHECKING,
    # Callable,
    Dict,
    List,
    Any,
    Union,
    # Mapping,
    # Sequence,
    # Tuple,
    # cast,
    Optional
)


ARROW_HEADER = "application/vnd.apache.arrow.file"
ARROW_FORMAT = "arrow"
JSON_HEADER = "application/json"
CUSTOM_JSON_FORMAT = "custom-json"
DEFAULT_RETRIES = 1

ARROW_HEADERS = {
    "Content-Type": ARROW_HEADER,
    "format": ARROW_FORMAT,
    "Accept": ARROW_HEADER,
}

JSON_HEADERS = {
    "Content-Type": JSON_HEADER,
    "Accept": JSON_HEADER,
}


class _AsyncConnection:
    def __init__(self, url):
        self._url = url
        self._async_client = AsyncClient()


def connect(url : str) -> _AsyncConnection:
    """
    Connects to a Wallaroo inference endpoint. This can be a cluster, authoritzed endpoint or an unauthenticated cluster or edge endpoint.
    """

    # TODO: Validate the URL and warm the HTTP connection cache with one query
    return _AsyncConnection(url)
    

def get_dataset_params(
    dataset: Optional[List[str]] = None,
    dataset_exclude: Optional[List[str]] = None,
    dataset_separator: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates the default set of parameters based on the fields of the dataset that are requested. This is pre-computed
    to reduce overhead during frequent inference requests.
    :param: dataset: Optional[List[str]] By default this is set to ["*"] which returns,
        ["time", "in", "out", "check_failures"]. Other available options - ["metadata"]
    :param: dataset_exclude: Optional[List[str]] If set, allows user to exclude parts of dataset.
    :param: dataset_separator: Optional[Union[Sequence[str], str]] If set to ".", return dataset will be flattened.
    """
    params = dict()
    default_dataset_exclude = ["metadata"]
    if dataset is not None:
        if "metadata" in dataset:
            default_dataset_exclude = []
    params["dataset[]"] = dataset or ["*"]  # type: ignore
    params["dataset.exclude[]"] = (
        [*dataset_exclude, *default_dataset_exclude]
        if dataset_exclude is not None
        else default_dataset_exclude
    )
    params["dataset.separator"] = dataset_separator or "."  # type: ignore
    return params


async def async_infer(
        connection: _AsyncConnection,
        tensor: Union[Dict[str, Any], pa.Table],
        headers: dict = None,
        dataset_params: Dict[str, Any] = None,
        timeout: Optional[Union[int, float]] = None,
        retries: Optional[int] = None,
    ):
    """
    Runs an async inference and returns an inference result on this deployment, given a tensor.
    :param: connection: Connection information constructed by `connect`.
    :param: tensor: Union[Dict[str, Any], pd.DataFrame, pa.Table] Inference data.
    :param: headers: Dict[str, Any] of headers to be passed to the inference endpoint, or None to use the default
    :param: dataset_params: Dict[str, Any] of parameters controlling the dataset return by the inference endpoint and computed by get_dataset_params(). Use get_dataset_params() to pre-computer the parameters for frequent inferences to reduce overhead or pass None and use the defaults.
    :param: timeout: Optional[Union[int, float]] infer requests will time out after
        the amount of seconds provided are exceeded. timeout defaults
        to 15 secs.
    :param: retries: Optional[int] Number of retries to use in case of Connection errors.
    :param: job_id: Optional[int] Job id to use for async inference.
    """
    timeout = _init_timeout(timeout)
    if isinstance(tensor, pa.Table):
        return await _async_infer_with_arrow(
            connection=connection,
            tensor=tensor,
            headers=headers or ARROW_HEADERS,
            timeout=timeout,
            params=dataset_params or get_dataset_params(),
            retries=retries,
        )
    elif isinstance(tensor, (dict, list)):
        return await _async_infer_with_json(
            connection=connection,
            tensor=tensor,
            headers=headers or JSON_HEADERS,
            timeout=timeout,
            params=dataset_params or get_dataset_params(),
            retries=retries,
        )
    else:
        raise TypeError(
            f"tensor is of type {type(tensor)} but 'pyarrow.Table', dict or list is required"
        )

async def _async_infer_with_json(
    connection: _AsyncConnection,
    tensor: Union[Dict[str, Any], List[Any]],
    headers: dict,
    timeout: Optional[Union[int, float]] = None,
    params: Optional[Dict[str, Any]] = None,
    retries: Optional[int] = None,
):
    response = await _make_async_infer_request(
        connection=connection,
        headers=headers,
        json_data=tensor,
        params=params,
        timeout=timeout,
        retries=retries,
    )
    try:
        data = response.json() if response is not None else None
    except (json.JSONDecodeError, ValueError) as err:
        raise ValueError("Infer response is not valid.") from err

    return data

async def _async_infer_with_arrow(
    connection: _AsyncConnection,
    tensor: pa.Table,
    headers: dict,
    timeout: Optional[Union[int, float]] = None,
    params: Optional[Dict[str, Any]] = None,
    retries: Optional[int] = None,
):
    input_arrow = _write_table_to_arrow_file(tensor, tensor.schema)
    response = await _make_async_infer_request(
        connection=connection,
        headers=headers,
        content=input_arrow.to_pybytes(),
        params=params,
        timeout=timeout,
        retries=retries,
    )

    try:
        with pa.ipc.open_file(response.content) as reader:
            data_table = reader.read_all()
    except (pa.ArrowInvalid, ValueError) as err:
        raise ValueError("Infer response is not valid.") from err
    return data_table

async def _make_async_infer_request(
    connection: _AsyncConnection,
    headers: Dict[str, str],
    content: Optional[Union[bytes, str]] = None,
    json_data: Optional[Any] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[int, float]] = None,
    retries: Optional[int] = None,
):

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(retries or DEFAULT_RETRIES),
        reraise=True,
        retry=retry_if_exception_type((TimeoutException, NetworkError)),
    ):
        with attempt:
            try:
                response = await connection._async_client.post(
                    connection._url,
                    content=content,
                    json=json_data,
                    params=params,
                    timeout=timeout,
                    headers=headers,
                )
                response.raise_for_status()
            except HTTPStatusError as status_err:
                raise status_err
    return response

def _write_table_to_arrow_file(table: pa.Table, schema: pa.Schema):
    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, schema) as arrow_ipc:
        arrow_ipc.write(table)
        arrow_ipc.close()
    return sink.getvalue()

def _init_timeout(timeout: Optional[Union[int, float]]) -> float:
    if timeout is None:
        timeout = 15
    if not isinstance(timeout, (int, float)):
        raise TypeError(
            f"timeout is {type(timeout)} but 'int' or 'float' is required"
        )
    return timeout

def flatten_np_array_columns(df, col):
    if isinstance(df[col][0], np.ndarray):
        return df[col].apply(lambda x: np.array(x).ravel())
    else:
        return df[col]

