

class BudgetExceededError(RuntimeError):
    """The run cannot start another model call within its spend limit."""

class ClientError(RuntimeError):
    """Raised when there is an error using a client."""

class InitializationError(RuntimeError):
    """Raised when there is an error during initialization."""

class V4APatchError(RuntimeError):
    """Raised when there is an error applying a V4A patch."""

