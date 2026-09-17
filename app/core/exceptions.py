"""Domain exceptions for OP-06 Triage."""


class TriageBaseException(Exception):
    """Base exception for all triage operations."""


class InvalidInputException(TriageBaseException):
    """Raised when text is binary, unparseable, or empty."""


class PolicyException(TriageBaseException):
    """Raised on policy lookup failures or configuration errors."""


class ModelInferenceException(TriageBaseException):
    """Raised when model inference fails."""
