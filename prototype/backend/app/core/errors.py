"""RFC 7807 Problem Details error formatting and exception handlers."""

from typing import Any, Optional
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class APIError(Exception):
    """Base application exception supporting RFC 7807 attributes."""

    def __init__(
        self,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        detail: str = "A business logic error occurred",
        title: Optional[str] = None,
        type_: str = "about:blank",
        headers: Optional[dict[str, str]] = None,
    ):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.title = title or _default_title_for_status(status_code)
        self.type_ = type_
        self.headers = headers or {}


def _default_title_for_status(code: int) -> str:
    titles = {
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        409: "Conflict",
        422: "Unprocessable Entity",
        500: "Internal Server Error",
    }
    return titles.get(code, "Error")


def rfc7807_problem_response(
    status_code: int,
    detail: str,
    title: Optional[str] = None,
    instance: Optional[str] = None,
    type_: str = "about:blank",
    headers: Optional[dict[str, str]] = None,
) -> JSONResponse:
    """Construct an RFC 7807 compliant JSONResponse."""
    content: dict[str, Any] = {
        "type": type_,
        "title": title or _default_title_for_status(status_code),
        "status": status_code,
        "detail": detail,
    }
    if instance:
        content["instance"] = instance

    return JSONResponse(
        status_code=status_code,
        content=content,
        media_type="application/problem+json",
        headers=headers,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI / Starlette HTTPExceptions with RFC 7807 format."""
    return rfc7807_problem_response(
        status_code=exc.status_code,
        detail=str(exc.detail),
        title=_default_title_for_status(exc.status_code),
        instance=request.url.path,
        headers=exc.headers,
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic validation errors with RFC 7807 format."""
    # Build detailed error representation
    errors = exc.errors()
    first_msg = errors[0]["msg"] if errors else "Validation error"
    return rfc7807_problem_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=first_msg,
        title="Unprocessable Entity",
        instance=request.url.path,
    )


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle custom application errors."""
    return rfc7807_problem_response(
        status_code=exc.status_code,
        detail=exc.detail,
        title=exc.title,
        instance=request.url.path,
        type_=exc.type_,
        headers=exc.headers,
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Fallback handler for unhandled exceptions."""
    return rfc7807_problem_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="An unexpected internal server error occurred.",
        title="Internal Server Error",
        instance=request.url.path,
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all RFC 7807 exception handlers on the FastAPI application."""
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
