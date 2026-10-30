

class BudgetExceededError(RuntimeError):
    """The run cannot start another model call within its spend limit."""

class ClientError(RuntimeError):
    """Raised when there is an error using a client."""

    def __init__(
        self, 
        msg: str = "", 
        *, 
        e: Exception | None = None
    ):
        """Keep the provider failure details.

        Args:
            msg: Failure message.
            e: Original provider error.
        """
        super().__init__(msg or str(e or "Client request failed."))
        self.status_code = getattr(e, "status_code", None)
        response = getattr(e, "response", None)
        self.headers = dict(response.headers) if response is not None else {}
        self.__cause__ = e


class CallCancelledError(RuntimeError):
    """Cancellation stopped a model call."""


class ReservationError(ValueError):
    """A reservation is invalid or has already been settled."""

class InitializationError(RuntimeError):
    """Raised when there is an error during initialization."""

class V4APatchError(RuntimeError):
    """Raised when there is an error applying a V4A patch."""

