import email_notifier
import pytest


class _StubSocket:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def sendall(self, data):
        pass

    def recv(self, bufsize):
        return b""


class _StubSocketModule:
    @staticmethod
    def create_connection(address, timeout=None):
        return _StubSocket()


@pytest.fixture(autouse=True)
def _stub_email_socket(monkeypatch):
    """The watcher sends a result email after every successful video, so the
    email socket must be stubbed for every test (otherwise fake-pipeline
    runs would dial the real email server)."""
    monkeypatch.setattr(email_notifier, "socket", _StubSocketModule())
