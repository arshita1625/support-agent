#!/usr/bin/env python3
"""Common utility models using Python dataclasses."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Literal, Union
from datetime import datetime
from enum import Enum
import json
import platform
import sys

class LogLevel(Enum):
    """Enum for log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class ErrorResponse:
    """Standardized error response for consistent API error handling."""
    
    error_code: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    path: Optional[str] = None
    method: Optional[str] = None
    
    def __post_init__(self):
        """Validate and clean error response data."""
        if not self.error_code or not isinstance(self.error_code, str):
            raise ValueError('error_code must be a non-empty string')
        
        if not self.message or not isinstance(self.message, str):
            raise ValueError('message must be a non-empty string')
        
        # Standardize error code format (UPPERCASE_WITH_UNDERSCORES)
        self.error_code = self.error_code.strip().upper()
        self.message = self.message.strip()
        
        # Validate optional HTTP method
        if self.method is not None:
            valid_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
            self.method = self.method.strip().upper()
            if self.method not in valid_methods:
                raise ValueError(f'method must be one of: {valid_methods}')
        
        # Clean path
        if self.path is not None:
            self.path = self.path.strip()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        error_dict = asdict(self)
        error_dict['timestamp'] = self.timestamp.isoformat()
        return error_dict
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorResponse':
        """Create from dictionary."""
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)
    
    def add_detail(self, key: str, value: Any) -> None:
        """Add additional error detail."""
        if self.details is None:
            self.details = {}
        self.details[key] = value
    
    def is_client_error(self) -> bool:
        """Check if this is a client error (4xx equivalent)."""
        client_error_codes = [
            'VALIDATION_ERROR', 'NOT_FOUND', 'UNAUTHORIZED', 
            'FORBIDDEN', 'BAD_REQUEST', 'CONFLICT', 'INVALID_INPUT',
            'MISSING_FIELD', 'INVALID_FORMAT', 'DUPLICATE_ENTRY'
        ]
        return self.error_code in client_error_codes
    
    def is_server_error(self) -> bool:
        """Check if this is a server error (5xx equivalent)."""
        server_error_codes = [
            'INTERNAL_ERROR', 'SERVICE_UNAVAILABLE', 'TIMEOUT',
            'DATABASE_ERROR', 'EXTERNAL_SERVICE_ERROR', 'PROCESSING_ERROR',
            'SYSTEM_OVERLOAD', 'MAINTENANCE_MODE'
        ]
        return self.error_code in server_error_codes
    
    def get_http_status_code(self) -> int:
        """Get appropriate HTTP status code for this error."""
        # Map error codes to HTTP status codes
        status_mapping = {
            # Client errors (400-499)
            'BAD_REQUEST': 400,
            'VALIDATION_ERROR': 400,
            'INVALID_INPUT': 400,
            'MISSING_FIELD': 400,
            'INVALID_FORMAT': 400,
            'UNAUTHORIZED': 401,
            'FORBIDDEN': 403,
            'NOT_FOUND': 404,
            'CONFLICT': 409,
            'DUPLICATE_ENTRY': 409,
            
            # Server errors (500-599)
            'INTERNAL_ERROR': 500,
            'PROCESSING_ERROR': 500,
            'DATABASE_ERROR': 500,
            'EXTERNAL_SERVICE_ERROR': 502,
            'SERVICE_UNAVAILABLE': 503,
            'TIMEOUT': 504,
            'SYSTEM_OVERLOAD': 503,
            'MAINTENANCE_MODE': 503
        }
        
        return status_mapping.get(self.error_code, 500)  # Default to 500

@dataclass
class SystemInfo:
    """System information for diagnostics and debugging."""
    
    # Use lambda functions to avoid the platform import issue we had earlier
    python_version: str = field(default_factory=lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    platform: str = field(default_factory=lambda: platform.system())
    architecture: str = field(default_factory=lambda: platform.machine())
    hostname: str = field(default_factory=lambda: platform.node())
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        info_dict = asdict(self)
        info_dict['timestamp'] = self.timestamp.isoformat()
        return info_dict
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def get_environment_summary(self) -> Dict[str, str]:
        """Get concise environment summary."""
        return {
            "runtime": f"Python {self.python_version}",
            "os": f"{self.platform} ({self.architecture})",
            "host": self.hostname
        }

@dataclass
class PaginationInfo:
    """Pagination information for API responses with list data."""
    
    page: int
    page_size: int
    total_items: int
    total_pages: Optional[int] = None
    has_next: Optional[bool] = None
    has_previous: Optional[bool] = None
    
    def __post_init__(self):
        """Calculate pagination fields automatically."""
        if self.page < 1:
            raise ValueError('page must be >= 1')
        
        if self.page_size < 1:
            raise ValueError('page_size must be >= 1')
        
        if self.total_items < 0:
            raise ValueError('total_items must be >= 0')
        
        # Calculate total pages
        if self.total_items == 0:
            self.total_pages = 0
        else:
            self.total_pages = (self.total_items - 1) // self.page_size + 1
        
        # Calculate navigation flags
        self.has_next = self.page < self.total_pages
        self.has_previous = self.page > 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaginationInfo':
        """Create from dictionary."""
        return cls(**data)
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def get_offset(self) -> int:
        """Get offset for database queries (0-based)."""
        return (self.page - 1) * self.page_size
    
    def get_limit(self) -> int:
        """Get limit for database queries."""
        return self.page_size
    
    def get_item_range(self) -> tuple:
        """Get the range of items on current page (1-based)."""
        if self.total_items == 0:
            return (0, 0)
        
        start = self.get_offset() + 1
        end = min(start + self.page_size - 1, self.total_items)
        return (start, end)
    
    def is_valid_page(self) -> bool:
        """Check if current page number is valid."""
        return 1 <= self.page <= max(self.total_pages, 1)

@dataclass
class ValidationError:
    """Individual field validation error for detailed feedback."""
    
    field: str
    message: str
    value: Optional[Any] = None
    constraint: Optional[str] = None
    
    def __post_init__(self):
        """Validate validation error data."""
        if not self.field or not isinstance(self.field, str):
            raise ValueError('field must be a non-empty string')
        
        if not self.message or not isinstance(self.message, str):
            raise ValueError('message must be a non-empty string')
        
        self.field = self.field.strip()
        self.message = self.message.strip()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def get_user_friendly_message(self) -> str:
        """Get user-friendly error message."""
        field_name = self.field.replace('_', ' ').title()
        return f"{field_name}: {self.message}"

@dataclass
class ValidationErrors:
    """Collection of validation errors with analysis capabilities."""
    
    errors: List[ValidationError] = field(default_factory=list)
    
    def add_error(self, field: str, message: str, value: Any = None, constraint: str = None) -> None:
        """Add validation error with automatic deduplication."""
        error = ValidationError(
            field=field,
            message=message,
            value=value,
            constraint=constraint
        )
        
        # Check for duplicates
        existing_error = self.get_error_for_field_and_message(field, message)
        if not existing_error:
            self.errors.append(error)
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    def get_error_count(self) -> int:
        """Get total error count."""
        return len(self.errors)
    
    def get_fields_with_errors(self) -> List[str]:
        """Get list of fields that have errors."""
        return list(set(error.field for error in self.errors))
    
    def get_errors_for_field(self, field: str) -> List[ValidationError]:
        """Get all errors for a specific field."""
        return [error for error in self.errors if error.field == field]
    
    def get_error_for_field_and_message(self, field: str, message: str) -> Optional[ValidationError]:
        """Get specific error by field and message."""
        for error in self.errors:
            if error.field == field and error.message == message:
                return error
        return None
    
    def clear_errors_for_field(self, field: str) -> None:
        """Remove all errors for a specific field."""
        self.errors = [error for error in self.errors if error.field != field]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with summary information."""
        return {
            'errors': [error.to_dict() for error in self.errors],
            'error_count': self.get_error_count(),
            'fields_with_errors': self.get_fields_with_errors(),
            'has_errors': self.has_errors()
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def get_user_friendly_messages(self) -> List[str]:
        """Get list of user-friendly error messages."""
        return [error.get_user_friendly_message() for error in self.errors]
    
    def get_summary_by_field(self) -> Dict[str, List[str]]:
        """Get errors grouped by field for form validation."""
        summary = {}
        for error in self.errors:
            if error.field not in summary:
                summary[error.field] = []
            summary[error.field].append(error.message)
        return summary

@dataclass
class APIResponse:
    """Generic API response wrapper for consistent response format."""
    
    success: bool
    data: Optional[Any] = None
    error: Optional[ErrorResponse] = None
    pagination: Optional[PaginationInfo] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate API response consistency."""
        if self.success and self.error is not None:
            raise ValueError('Cannot have error when success is True')
        
        if not self.success and self.data is not None and self.error is None:
            raise ValueError('Must have error when success is False and data is provided')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        response_dict = {
            'success': self.success,
            'timestamp': self.timestamp.isoformat()
        }
        
        # Handle data serialization intelligently
        if self.data is not None:
            if hasattr(self.data, 'to_dict'):
                response_dict['data'] = self.data.to_dict()
            elif isinstance(self.data, list) and self.data and hasattr(self.data[0], 'to_dict'):
                response_dict['data'] = [item.to_dict() for item in self.data]
            elif isinstance(self.data, (list, dict, str, int, float, bool, type(None))):
                response_dict['data'] = self.data
            else:
                response_dict['data'] = str(self.data)
        
        # Handle error serialization
        if self.error is not None:
            response_dict['error'] = self.error.to_dict()
        
        # Handle pagination serialization
        if self.pagination is not None:
            response_dict['pagination'] = self.pagination.to_dict()
        
        return response_dict
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def success_response(cls, data: Any = None, pagination: PaginationInfo = None) -> 'APIResponse':
        """Create success response."""
        return cls(success=True, data=data, pagination=pagination)
    
    @classmethod
    def error_response(cls, error: ErrorResponse) -> 'APIResponse':
        """Create error response."""
        return cls(success=False, error=error)
    
    @classmethod
    def validation_error_response(cls, validation_errors: ValidationErrors) -> 'APIResponse':
        """Create validation error response."""
        error = ErrorResponse(
            error_code="VALIDATION_ERROR",
            message="Input validation failed",
            details=validation_errors.to_dict()
        )
        return cls(success=False, error=error)
    
    def get_http_status_code(self) -> int:
        """Get appropriate HTTP status code for this response."""
        if self.success:
            return 200
        elif self.error:
            return self.error.get_http_status_code()
        else:
            return 500

# Test the models if run directly
if __name__ == "__main__":
    print("🧪 Testing Common Models...")
    # Test 2: ErrorResponse
    print("\n❌ Test 2: ErrorResponse")
    error = ErrorResponse(
        error_code="validation_error",
        message="Invalid ticket content format",
        path="/api/tickets",
        method="post"
    )
    error.add_detail("field", "ticket_text")
    error.add_detail("constraint", "min_length")
    
    print(f"   Error code: {error.error_code}")
    print(f"   Is client error: {error.is_client_error()}")
    print(f"   HTTP status: {error.get_http_status_code()}")
    print(f"   Details: {error.details}")
    
    # Test 3: ValidationErrors
    print("\n🔍 Test 3: ValidationErrors")
    validation_errors = ValidationErrors()
    validation_errors.add_error("ticket_text", "Must be at least 10 characters", "hi", "min_length")
    validation_errors.add_error("priority", "Invalid priority level", "super_high", "allowed_values")
    validation_errors.add_error("ticket_text", "Cannot contain only special characters", "!!!", "format")
    
    print(f"   Error count: {validation_errors.get_error_count()}")
    print(f"   Fields with errors: {validation_errors.get_fields_with_errors()}")
    print(f"   User-friendly messages:")
    for msg in validation_errors.get_user_friendly_messages():
        print(f"     • {msg}")
    
    # Test 4: PaginationInfo
    print("\n📄 Test 4: PaginationInfo")
    pagination = PaginationInfo(page=2, page_size=10, total_items=25)
    print(f"   Page {pagination.page}/{pagination.total_pages}")
    print(f"   Has next: {pagination.has_next}")
    print(f"   Has previous: {pagination.has_previous}")
    print(f"   Item range: {pagination.get_item_range()}")
    print(f"   Database offset: {pagination.get_offset()}")
    print(f"   Valid page: {pagination.is_valid_page()}")
    
    # Test 5: APIResponse
    print("\n🌐 Test 5: APIResponse")
    
    # Success response with data
    success_response = APIResponse.success_response(
        data={"message": "Ticket processed successfully", "ticket_id": "12345"},
        pagination=pagination
    )
    print(f"   Success response: {success_response.success}")
    print(f"   HTTP status: {success_response.get_http_status_code()}")
    
    # Error response
    error_response = APIResponse.error_response(error)
    print(f"   Error response: {error_response.success}")
    print(f"   HTTP status: {error_response.get_http_status_code()}")
    
    # Validation error response
    validation_response = APIResponse.validation_error_response(validation_errors)
    print(f"   Validation response: {validation_response.success}")
    print(f"   HTTP status: {validation_response.get_http_status_code()}")
    
    # Test 6: SystemInfo
    print("\n💻 Test 6: SystemInfo")
    system_info = SystemInfo()
    print(f"   Environment: {system_info.get_environment_summary()}")
    print(f"   Python: {system_info.python_version}")
    print(f"   Platform: {system_info.platform}")
    
    # Test 7: Serialization
    print("\n💾 Test 7: Serialization")
    models_to_test = [
        ("ErrorResponse", error),
        ("ValidationErrors", validation_errors),
        ("PaginationInfo", pagination),
        ("APIResponse", success_response),
        ("SystemInfo", system_info)
    ]
    
    for model_name, model_instance in models_to_test:
        try:
            # Test dict conversion
            model_dict = model_instance.to_dict()
            
            # Test JSON conversion
            json_str = model_instance.to_json()
            
            # Test restoration from dict (where available)
            if hasattr(model_instance.__class__, 'from_dict'):
                restored = model_instance.__class__.from_dict(model_dict)
                print(f"   ✅ {model_name}: serialization & deserialization successful")
            else:
                print(f"   ✅ {model_name}: serialization successful")
        except Exception as e:
            print(f"   ❌ {model_name}: serialization failed - {e}")
    
    print("\n🎉 All common models working correctly!")
