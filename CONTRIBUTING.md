# Contributing to Music Theory AI Agent

Thank you for your interest in contributing to the Music Theory AI Agent project! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/music-theory-ai-agent.git
   cd music-theory-ai-agent
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]
   ```

## Development Setup

### Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pytest**: Testing

Run these before committing:
```bash
black .
flake8 .
mypy .
pytest
```

### Pre-commit Hooks

Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

## Making Changes

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure tests pass:
   ```bash
   pytest
   ```

3. Commit your changes:
   ```bash
   git add .
   git commit -m "Add your feature description"
   ```

4. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

5. Create a Pull Request on GitHub

## Types of Contributions

### Bug Reports

When reporting bugs, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs

### Feature Requests

For new features, please:
- Describe the use case
- Explain why it would be useful
- Consider implementation complexity
- Provide examples if possible

### Code Contributions

We welcome contributions in these areas:
- New music analysis tools
- Performance improvements
- Documentation improvements
- Test coverage
- Bug fixes

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_specific.py
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies (APIs, file I/O)

### Test Structure

```python
def test_function_name():
    """Test description."""
    # Arrange
    input_data = "test input"
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_output
```

## Documentation

### Code Documentation

- Use docstrings for all public functions/classes
- Follow Google docstring format
- Include type hints where possible

### README Updates

- Update README.md for significant changes
- Include usage examples
- Update installation instructions if needed

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add tests for new functionality
4. Follow the existing code style
5. Write clear commit messages
6. Request review from maintainers

### PR Template

When creating a PR, please include:
- Description of changes
- Related issues
- Testing performed
- Screenshots (if applicable)

## Release Process

Releases are managed by maintainers:
1. Update version in `setup.py`
2. Update `CHANGELOG.md`
3. Create GitHub release
4. Publish to PyPI (if applicable)

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to create a welcoming environment for all contributors.

## Questions?

Feel free to open an issue for questions about contributing or the project in general.
