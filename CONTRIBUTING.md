# Contributing to ROCm Patch Repository

First off, thank you for considering contributing to the ROCm Patch Repository! This project helps the AMD GPU community solve real-world problems.

## Code of Conduct

Be respectful, professional, and inclusive. We're all here to make ROCm better.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **System information** (OS, ROCm version, GPU model)
- **Steps to reproduce**
- **Expected vs actual behavior**
- **Error messages and logs**

### Suggesting Features

Feature suggestions should address real ROCm issues. Include:

- **Problem description**
- **Proposed solution**
- **Alternative approaches**
- **Link to ROCm GitHub issue if applicable**

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Follow naming conventions**: `feature/description` or `fix/description`
3. **Write tests** for your patch
4. **Update documentation** as needed
5. **Follow code style** (Black, Flake8, Pylint)
6. **Write clear commit messages**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/rocm-patch.git
cd rocm-patch

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install development tools
pip install pre-commit
pre-commit install

# Run tests
pytest tests/
```

## Patch Development Guidelines

### Creating a New Patch

1. **Research the issue** thoroughly
2. **Document the problem** in `docs/patches/`
3. **Create patch module** in `src/patches/`
4. **Write tests** in `tests/`
5. **Add patch metadata** to registry

### Patch Structure

```python
"""
Patch Name
~~~~~~~~~~

Description of what this patch fixes.

**Problem**: Detailed problem description
**Solution**: How this patch fixes it
**Affected**: ROCm versions, GPU architectures
**Status**: Tested | Beta | Experimental
"""

class PatchName:
    """Patch implementation."""
    
    def apply(self):
        """Apply the patch with proper error handling."""
        pass
    
    def verify(self):
        """Verify patch was applied correctly."""
        pass
    
    def rollback(self):
        """Safely rollback the patch."""
        pass
```

### Code Quality Standards

- **Python**: Black (line length 88), Flake8, Pylint, MyPy
- **Comments**: Docstrings for all public functions/classes
- **Error Handling**: Comprehensive with informative messages
- **Logging**: Use logging module, not print statements
- **Testing**: >80% coverage for new code

### Testing Requirements

All patches must include:

1. **Unit tests**: Test individual components
2. **Integration tests**: Test on actual system (when safe)
3. **Boundary tests**: Edge cases and error conditions
4. **Performance tests**: Measure overhead

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test category
pytest tests/unit/ -v
pytest tests/integration/ -v
```

## Commit Message Guidelines

Format: `type(scope): description`

Types:
- `feat`: New feature or patch
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

Examples:
```
feat(patches): add memory access fault fix for ROCm 7.1
fix(utils): correct GPU architecture detection for gfx1100
docs(readme): update installation instructions
test(patches): add boundary tests for VRAM allocation
```

## Documentation Standards

- **README**: Clear installation and usage
- **Patch docs**: Problem, solution, usage, examples
- **API docs**: Docstrings for all public APIs
- **Comments**: Explain "why", not "what"

## Review Process

1. **Automated checks**: CI/CD must pass
2. **Code review**: At least one maintainer approval
3. **Testing**: Verify on actual hardware when possible
4. **Documentation**: Must be complete and accurate

## Community

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Pull Requests**: Code contributions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue or discussion if you have questions!

---

Thank you for contributing! 🎉
