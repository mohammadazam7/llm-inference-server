import llm_inference_server


def test_version_is_set() -> None:
    assert llm_inference_server.__version__
