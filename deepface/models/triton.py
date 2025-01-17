from tritonclient.grpc import InferenceServerClient


class TritonClientManager:
    def __init__(self):
        self.client = InferenceServerClient(
            url="localhost:8001",
            verbose=False,
            ssl=False,
        )

    def __enter__(self):
        return self.client

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()


def triton_client():
    return TritonClientManager()