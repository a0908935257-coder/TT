"""
TLS/SSL Configuration Module.

Provides secure TLS configuration for Redis connections,
internal API communication, and certificate management.
"""

import ssl
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core import get_logger

logger = get_logger(__name__)


class TLSVersion(Enum):
    """Supported TLS versions."""

    TLS_1_2 = "TLSv1.2"
    TLS_1_3 = "TLSv1.3"


class CertificateType(Enum):
    """Certificate type."""

    CA = "ca"  # Certificate Authority
    SERVER = "server"
    CLIENT = "client"


@dataclass
class CertificateInfo:
    """Certificate information."""

    path: Path
    cert_type: CertificateType
    common_name: str
    issuer: str
    valid_from: datetime
    valid_until: datetime
    fingerprint: str
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": str(self.path),
            "cert_type": self.cert_type.value,
            "common_name": self.common_name,
            "issuer": self.issuer,
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat(),
            "fingerprint": self.fingerprint,
            "is_valid": self.is_valid,
        }

    def is_expired(self) -> bool:
        """Check if certificate is expired."""
        return datetime.now(timezone.utc) > self.valid_until

    def days_until_expiry(self) -> int:
        """Get days until certificate expires."""
        delta = self.valid_until - datetime.now(timezone.utc)
        return max(0, delta.days)


@dataclass
class TLSConfig:
    """TLS configuration settings."""

    # Enabled
    enabled: bool = True

    # Version
    min_version: TLSVersion = TLSVersion.TLS_1_2
    max_version: TLSVersion = TLSVersion.TLS_1_3

    # Certificate paths
    ca_cert_path: Optional[Path] = None
    server_cert_path: Optional[Path] = None
    server_key_path: Optional[Path] = None
    client_cert_path: Optional[Path] = None
    client_key_path: Optional[Path] = None

    # Verification
    verify_mode: ssl.VerifyMode = ssl.CERT_REQUIRED
    check_hostname: bool = True

    # Cipher suites (TLS 1.2)
    ciphers: str = (
        "ECDHE+AESGCM:DHE+AESGCM:ECDHE+CHACHA20:DHE+CHACHA20:"
        "!aNULL:!MD5:!DSS:!RC4:!3DES"
    )

    # Certificate expiry warning (days)
    expiry_warning_days: int = 30


class TLSManager:
    """
    Manages TLS/SSL configurations and certificates.

    Provides secure SSL contexts for Redis, HTTP clients,
    and internal service communication.
    """

    def __init__(self, config: Optional[TLSConfig] = None):
        """
        Initialize TLS manager.

        Args:
            config: TLS configuration
        """
        self._config = config or TLSConfig()
        self._certificates: Dict[str, CertificateInfo] = {}
        self._ssl_contexts: Dict[str, ssl.SSLContext] = {}

    @property
    def enabled(self) -> bool:
        """Check if TLS is enabled."""
        return self._config.enabled

    def create_redis_ssl_context(self) -> Optional[ssl.SSLContext]:
        """
        Create SSL context for Redis connections.

        Returns:
            SSL context configured for Redis or None if disabled
        """
        if not self._config.enabled:
            return None

        context = self._create_base_context(ssl.Purpose.SERVER_AUTH)

        # Load CA certificate if provided
        if self._config.ca_cert_path and self._config.ca_cert_path.exists():
            context.load_verify_locations(cafile=str(self._config.ca_cert_path))

        # Load client certificate for mutual TLS
        if (
            self._config.client_cert_path
            and self._config.client_key_path
            and self._config.client_cert_path.exists()
            and self._config.client_key_path.exists()
        ):
            context.load_cert_chain(
                certfile=str(self._config.client_cert_path),
                keyfile=str(self._config.client_key_path),
            )

        self._ssl_contexts["redis"] = context
        logger.info("Redis SSL context created")
        return context

    def create_server_ssl_context(self) -> Optional[ssl.SSLContext]:
        """
        Create SSL context for server (accepting connections).

        Returns:
            SSL context for server or None if disabled
        """
        if not self._config.enabled:
            return None

        if not self._config.server_cert_path or not self._config.server_key_path:
            raise ValueError("Server certificate and key required")

        context = self._create_base_context(ssl.Purpose.CLIENT_AUTH)

        # Load server certificate
        context.load_cert_chain(
            certfile=str(self._config.server_cert_path),
            keyfile=str(self._config.server_key_path),
        )

        # Load CA for client verification (mutual TLS)
        if self._config.ca_cert_path and self._config.ca_cert_path.exists():
            context.load_verify_locations(cafile=str(self._config.ca_cert_path))
            context.verify_mode = self._config.verify_mode

        self._ssl_contexts["server"] = context
        logger.info("Server SSL context created")
        return context

    def create_client_ssl_context(
        self,
        verify: bool = True,
    ) -> Optional[ssl.SSLContext]:
        """
        Create SSL context for client (making connections).

        Args:
            verify: Whether to verify server certificate

        Returns:
            SSL context for client or None if disabled
        """
        if not self._config.enabled:
            return None

        context = self._create_base_context(ssl.Purpose.SERVER_AUTH)

        if verify:
            # Load CA certificate
            if self._config.ca_cert_path and self._config.ca_cert_path.exists():
                context.load_verify_locations(cafile=str(self._config.ca_cert_path))
            else:
                # Use system CA certificates
                context.load_default_certs()

            context.verify_mode = ssl.CERT_REQUIRED
            context.check_hostname = self._config.check_hostname
        else:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

        # Load client certificate for mutual TLS
        if (
            self._config.client_cert_path
            and self._config.client_key_path
            and self._config.client_cert_path.exists()
            and self._config.client_key_path.exists()
        ):
            context.load_cert_chain(
                certfile=str(self._config.client_cert_path),
                keyfile=str(self._config.client_key_path),
            )

        self._ssl_contexts["client"] = context
        logger.info("Client SSL context created")
        return context

    def _create_base_context(self, purpose: ssl.Purpose) -> ssl.SSLContext:
        """Create base SSL context with common settings."""
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT if purpose == ssl.Purpose.SERVER_AUTH else ssl.PROTOCOL_TLS_SERVER)

        # Set minimum TLS version
        if self._config.min_version == TLSVersion.TLS_1_2:
            context.minimum_version = ssl.TLSVersion.TLSv1_2
        else:
            context.minimum_version = ssl.TLSVersion.TLSv1_3

        # Set maximum TLS version
        if self._config.max_version == TLSVersion.TLS_1_3:
            context.maximum_version = ssl.TLSVersion.TLSv1_3
        else:
            context.maximum_version = ssl.TLSVersion.TLSv1_2

        # Set cipher suites (only for TLS 1.2)
        if self._config.min_version == TLSVersion.TLS_1_2:
            context.set_ciphers(self._config.ciphers)

        return context

    def load_certificate(
        self,
        path: Path,
        cert_type: CertificateType,
    ) -> Optional[CertificateInfo]:
        """
        Load and parse certificate information.

        Args:
            path: Path to certificate file
            cert_type: Type of certificate

        Returns:
            Certificate information or None if failed
        """
        if not path.exists():
            logger.error(f"Certificate not found: {path}")
            return None

        try:
            import hashlib

            with open(path, "rb") as f:
                cert_data = f.read()

            # Calculate fingerprint
            fingerprint = hashlib.sha256(cert_data).hexdigest()

            # Try to parse with cryptography if available
            try:
                from cryptography import x509
                from cryptography.hazmat.backends import default_backend

                cert = x509.load_pem_x509_certificate(cert_data, default_backend())

                info = CertificateInfo(
                    path=path,
                    cert_type=cert_type,
                    common_name=self._get_cn(cert.subject),
                    issuer=self._get_cn(cert.issuer),
                    valid_from=cert.not_valid_before_utc,
                    valid_until=cert.not_valid_after_utc,
                    fingerprint=fingerprint[:32],
                    is_valid=not self._is_cert_expired(cert),
                )

            except ImportError:
                # Fallback without cryptography library
                info = CertificateInfo(
                    path=path,
                    cert_type=cert_type,
                    common_name="unknown",
                    issuer="unknown",
                    valid_from=datetime.now(timezone.utc),
                    valid_until=datetime.now(timezone.utc),
                    fingerprint=fingerprint[:32],
                    is_valid=True,
                )

            self._certificates[str(path)] = info

            # Check expiry warning
            if info.days_until_expiry() <= self._config.expiry_warning_days:
                logger.warning(
                    f"Certificate {path} expires in {info.days_until_expiry()} days"
                )

            return info

        except Exception as e:
            logger.error(f"Failed to load certificate {path}: {e}")
            return None

    def _get_cn(self, name) -> str:
        """Extract Common Name from certificate name."""
        try:
            from cryptography.x509.oid import NameOID

            cn = name.get_attributes_for_oid(NameOID.COMMON_NAME)
            return cn[0].value if cn else "unknown"
        except Exception:
            return "unknown"

    def _is_cert_expired(self, cert) -> bool:
        """Check if certificate is expired."""
        try:
            return datetime.now(timezone.utc) > cert.not_valid_after_utc
        except Exception:
            return False

    def get_certificate_info(self, path: Path) -> Optional[CertificateInfo]:
        """Get cached certificate information."""
        return self._certificates.get(str(path))

    def list_certificates(self) -> List[CertificateInfo]:
        """List all loaded certificates."""
        return list(self._certificates.values())

    def check_certificates_expiry(self) -> List[CertificateInfo]:
        """
        Check for certificates nearing expiry.

        Returns:
            List of certificates within warning threshold
        """
        expiring = []
        for cert in self._certificates.values():
            if cert.days_until_expiry() <= self._config.expiry_warning_days:
                expiring.append(cert)
        return expiring

    def validate_certificate_chain(
        self,
        cert_path: Path,
        ca_path: Path,
    ) -> Tuple[bool, str]:
        """
        Validate certificate against CA.

        Args:
            cert_path: Path to certificate
            ca_path: Path to CA certificate

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            from cryptography import x509
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives import hashes
            from cryptography.x509 import verification

            with open(cert_path, "rb") as f:
                cert = x509.load_pem_x509_certificate(f.read(), default_backend())

            with open(ca_path, "rb") as f:
                ca_cert = x509.load_pem_x509_certificate(f.read(), default_backend())

            # Verify signature
            try:
                ca_cert.public_key().verify(
                    cert.signature,
                    cert.tbs_certificate_bytes,
                    cert.signature_algorithm_parameters,
                )
                return True, "Certificate chain valid"
            except Exception as e:
                return False, f"Signature verification failed: {e}"

        except ImportError:
            return True, "Validation skipped (cryptography not available)"
        except Exception as e:
            return False, f"Validation error: {e}"


class SecureConnectionFactory:
    """Factory for creating secure connections."""

    def __init__(self, tls_manager: TLSManager):
        """
        Initialize factory.

        Args:
            tls_manager: TLS manager instance
        """
        self._tls = tls_manager

    def get_redis_kwargs(self) -> Dict[str, Any]:
        """
        Get kwargs for secure Redis connection.

        Returns:
            Dictionary of Redis connection kwargs
        """
        if not self._tls.enabled:
            return {}

        ssl_context = self._tls.create_redis_ssl_context()
        if ssl_context:
            return {"ssl": ssl_context}
        return {}

    def get_aiohttp_kwargs(self) -> Dict[str, Any]:
        """
        Get kwargs for secure aiohttp client.

        Returns:
            Dictionary of aiohttp connector kwargs
        """
        if not self._tls.enabled:
            return {}

        ssl_context = self._tls.create_client_ssl_context()
        if ssl_context:
            return {"ssl": ssl_context}
        return {}

    def get_websocket_kwargs(self) -> Dict[str, Any]:
        """
        Get kwargs for secure WebSocket connection.

        Returns:
            Dictionary of WebSocket connection kwargs
        """
        if not self._tls.enabled:
            return {}

        ssl_context = self._tls.create_client_ssl_context()
        if ssl_context:
            return {"ssl": ssl_context}
        return {}


# Convenience function for environment-based configuration
def create_tls_config_from_env() -> TLSConfig:
    """
    Create TLS configuration from environment variables.

    Environment variables:
        TLS_ENABLED: Enable TLS (default: true)
        TLS_MIN_VERSION: Minimum TLS version (TLSv1.2 or TLSv1.3)
        TLS_CA_CERT: Path to CA certificate
        TLS_SERVER_CERT: Path to server certificate
        TLS_SERVER_KEY: Path to server private key
        TLS_CLIENT_CERT: Path to client certificate
        TLS_CLIENT_KEY: Path to client private key
        TLS_VERIFY_MODE: Verification mode (required, optional, none)
        TLS_CHECK_HOSTNAME: Check hostname (default: true)

    Returns:
        TLSConfig instance
    """
    enabled = os.environ.get("TLS_ENABLED", "true").lower() == "true"

    min_version_str = os.environ.get("TLS_MIN_VERSION", "TLSv1.2")
    min_version = (
        TLSVersion.TLS_1_3 if min_version_str == "TLSv1.3" else TLSVersion.TLS_1_2
    )

    verify_mode_str = os.environ.get("TLS_VERIFY_MODE", "required").lower()
    verify_mode = {
        "required": ssl.CERT_REQUIRED,
        "optional": ssl.CERT_OPTIONAL,
        "none": ssl.CERT_NONE,
    }.get(verify_mode_str, ssl.CERT_REQUIRED)

    return TLSConfig(
        enabled=enabled,
        min_version=min_version,
        ca_cert_path=Path(p) if (p := os.environ.get("TLS_CA_CERT")) else None,
        server_cert_path=Path(p) if (p := os.environ.get("TLS_SERVER_CERT")) else None,
        server_key_path=Path(p) if (p := os.environ.get("TLS_SERVER_KEY")) else None,
        client_cert_path=Path(p) if (p := os.environ.get("TLS_CLIENT_CERT")) else None,
        client_key_path=Path(p) if (p := os.environ.get("TLS_CLIENT_KEY")) else None,
        verify_mode=verify_mode,
        check_hostname=os.environ.get("TLS_CHECK_HOSTNAME", "true").lower() == "true",
    )
