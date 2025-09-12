#!/usr/bin/env python3
"""Common utility models using Python dataclasses."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Literal, Union
from datetime import datetime
from enum import Enum
import json
import platform
import sys

class StatusLevel(Enum):
    """Enum for system status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class LogLevel(Enum):
    """Enum for log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class HealthStatus:
    """System health check response."""
    
    status: Literal["healthy", "degraded", "unhealthy"]
    services: Dict[str, bool] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    uptime_seconds: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    def __post_init__(self):
        """Validate health status."""
        valid_statuses = ["healthy", "degraded", "unhealthy"]
        if self.status not in valid_statuses:
            raise ValueError(f'status must be one of: {valid_statuses}')
        
        # Validate services
        if not isinstance(self.services, dict):
            raise ValueError('services must be a dictionary')
        
        for service_name, service_status in self.services.items():
            if not isinstance(service_name, str):
                raise ValueError('Service names must be strings')
            if not isinstance(service_status, bool):
                raise ValueError('Service statuses must be boolean')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        health_dict = asdict(self)
        health_dict['timestamp'] = self.timestamp.isoformat()
        return health_dict
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HealthStatus':
        """Create from dictionary."""
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)
    
    def add_service_status(self, service_name: str, is_healthy: bool) -> None:
        """Add service health status."""
        if not isinstance(service_name, str) or not service_name.strip():
            raise ValueError('Service name must be a non-empty string')
        if not isinstance(is_healthy, bool):
            raise ValueError('Service status must be boolean')
        
        self.services[service_name.strip()] = is_healthy
    
    def is_all_services_healthy(self) -> bool:
        """Check if all services are healthy."""
        return all(self.services.values()) if self.services else True
    
    def get_unhealthy_services(self) -> List[str]:
        """Get list of unhealthy services."""
        return [name for name, status in self.services.items() if not status]

@dataclass
class ErrorResponse:
    """Standardized error response."""
    
    error_code: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    path: Optional[str] = None
    method: Optional[str] = None
    
    def __post_init__(self):
        """Validate error response."""
        if not self.error_code or not isinstance(self.error_code, str):
            raise ValueError('error_code must be a non-empty string')
        
        if not self.message or not isinstance(self.message, str):
            raise ValueError('message must be a non-empty string')
        
        self.error_code = self.error_code.strip().upper()
        self.message = self.message.strip()
        
        # Validate optional fields
        if self.path is not None:
            self.path = self.path.strip()
        
        if self.method is not None:
            valid_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
            self.method = self.method.strip().upper()
            if self.method not in valid_methods:
                raise ValueError(f'method must be one of: {valid_methods}')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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

@dataclass
class SystemInfo:
    """System information for diagnostics."""
    
    # FIX: Use lambda functions for platform calls
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

@dataclass
class PaginationInfo:
    """Pagination information for API responses."""
    
    page: int
    page_size: int
    total_items: int
    total_pages: Optional[int] = None
    has_next: Optional[bool] = None
    has_previous: Optional[bool] = None
    
    def __post_init__(self):
        """Calculate pagination fields."""
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
        
        # Calculate has_next and has_previous
        self.has_next = self.page < self.total_pages
        self.has_previous = self.page > 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

@dataclass
class ValidationError:
    """Individual validation error."""
    
    field: str
    message: str
    value: Optional[Any] = None
    constraint: Optional[str] = None
    
    def __post_init__(self):
        """Validate validation error."""
        if not self.field or not isinstance(self.field, str):
            raise ValueError('field must be a non-empty string')
        
        if not self.message or not isinstance(self.message, str):
            raise ValueError('message must be a non-empty string')
        
        self.field = self.field.strip()
        self.message = self.message.strip()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

@dataclass
class ValidationErrors:
    """Collection of validation errors."""
    
    errors: List[ValidationError] = field(default_factory=list)
    
    def add_error(self, field: str, message: str, value: Any = None, constraint: str = None) -> None:
        """Add validation error."""
        error = ValidationError(
            field=field,
            message=message,
            value=value,
            constraint=constraint
        )
        self.errors.append(error)
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'errors': [error.to_dict() for error in self.errors],
            'error_count': len(self.errors),
            'fields_with_errors': [error.field for error in self.errors]
        }

@dataclass
class APIResponse:
    """Generic API response wrapper."""
    
    success: bool
    data: Optional[Any] = None
    error: Optional[ErrorResponse] = None
    pagination: Optional[PaginationInfo] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate API response."""
        if self.success and self.error is not None:
            raise ValueError('Cannot have error when success is True')
        
        if not self.success and self.data is not None and self.error is None:
            raise ValueError('Must have error when success is False')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        response_dict = {
            'success': self.success,
            'timestamp': self.timestamp.isoformat()
        }
        
        if self.data is not None:
            # Handle different data types
            if hasattr(self.data, 'to_dict'):
                response_dict['data'] = self.data.to_dict()
            elif isinstance(self.data, (list, dict, str, int, float, bool)):
                response_dict['data'] = self.data
            else:
                response_dict['data'] = str(self.data)
        
        if self.error is not None:
            response_dict['error'] = self.error.to_dict()
        
        if self.pagination is not None:
            response_dict['pagination'] = self.pagination.to_dict()
        
        return response_dict
    
    @classmethod 
    def success_response(cls, data: Any = None, pagination: PaginationInfo = None) -> 'APIResponse':
        """Create success response."""
        return cls(success=True, data=data, pagination=pagination)
    
    @classmethod
    def error_response(cls, error: ErrorResponse) -> 'APIResponse':
        """Create error response."""
        return cls(success=False, error=error)

# Minimal test
if __name__ == "__main__":
    print("🧪 Testing Common Models...")
    
    # Test SystemInfo (this was failing before)
    system_info = SystemInfo()
    print(f"✅ SystemInfo: Python {system_info.python_version} on {system_info.platform}")
    print(f"   Architecture: {system_info.architecture}")
    print(f"   Hostname: {system_info.hostname}")
    
    # Test other models
    health = HealthStatus(status="healthy")
    error = ErrorResponse(error_code="TEST", message="Test error")
    pagination = PaginationInfo(page=1, page_size=10, total_items=25)
    
    print("✅ All common models work!")
