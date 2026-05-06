import pytest
from nexus.utils.secret_scanner import scan_for_secrets, scan_content_safe, SecretMatch

@pytest.mark.safety
def test_aws_access_key_detected():
    content = "AWS_KEY = 'AKIAIOSFODNN7EXAMPLE'"
    is_safe, matches = scan_content_safe(content)
    assert is_safe is False
    assert any("AWS" in m.pattern_name for m in matches)

@pytest.mark.safety
def test_github_token_detected():
    content = "token = 'ghp_abc123defghijklmnopqrstuvwxyz1234'"
    is_safe, matches = scan_content_safe(content)
    assert is_safe is False
    assert any("GitHub" in m.pattern_name for m in matches)

@pytest.mark.safety
def test_generic_password_detected():
    content = "password = 'mysupersecretpass123'"
    is_safe, matches = scan_content_safe(content)
    assert is_safe is False
    assert any("Password" in m.pattern_name for m in matches)

@pytest.mark.safety
def test_private_key_header_detected():
    content = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA..."
    is_safe, matches = scan_content_safe(content)
    assert is_safe is False
    assert any("Private Key" in m.pattern_name for m in matches)

@pytest.mark.safety
def test_database_url_with_credentials_detected():
    content = "DB_URL = 'postgresql://admin:s3cr3tpassword@localhost/mydb'"
    is_safe, matches = scan_content_safe(content)
    assert is_safe is False
    assert any("Database" in m.pattern_name for m in matches)

@pytest.mark.safety
def test_redacted_value_does_not_contain_full_secret():
    secret = "sk-abcdefghijklmnop123456789"
    content = f"API_KEY = '{secret}'"
    matches = scan_for_secrets(content)
    for m in matches:
        # Redacted value should be first 4 chars + **** (length 8)
        assert len(m.redacted_value) <= 8
        assert m.redacted_value != secret
        assert m.redacted_value.startswith(secret[:4])

@pytest.mark.safety
def test_clean_python_file_passes():
    content = """
class Calculator:
    def add(self, a, b):
        return a + b
    """
    is_safe, matches = scan_content_safe(content)
    assert is_safe is True
    assert len(matches) == 0

@pytest.mark.safety
def test_scanner_does_not_raise_on_binary_content():
    # Null bytes represent binary content
    content = "some text\x00\x01\x02"
    is_safe, matches = scan_content_safe(content)
    assert is_safe is False
    assert matches == []

@pytest.mark.safety
def test_multiple_secrets_all_detected():
    content = """
    AWS_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE
    DB_URL=postgresql://user:pass@host/db
    GITHUB_TOKEN=ghp_123456789012345678901234567890123456
    """
    is_safe, matches = scan_content_safe(content)
    assert is_safe is False
    # Should find at least 3 matches
    assert len(matches) >= 3

@pytest.mark.safety
def test_line_numbers_are_accurate():
    content = "line1\nline2\nAPI_KEY='secret12345678'\nline4"
    matches = scan_for_secrets(content)
    # The API_KEY is on line 3
    assert any(m.line_number == 3 for m in matches)
