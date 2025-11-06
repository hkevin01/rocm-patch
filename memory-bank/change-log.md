# Change Log

## [Unreleased] - 2025-11-06

### Project Initialization
**Date**: 2025-11-06  
**Component**: Project Structure  
**Changes**:
- Initialized ROCm Patch Repository with comprehensive structure
- Created memory-bank documentation system
- Set up src layout with patches, utils, and tests directories
- Established GitHub workflows and templates
- Added .vscode configuration for development

**Testing Notes**: N/A - Initial setup  
**Contributors**: Project Maintainers

---

## Version History

### [0.1.0] - TBD
**Planned Features**:
- Initial patch collection for ROCm 7.1.x
- Memory access fault patches
- VRAM allocation fixes
- Basic installation script
- Docker testing environment

---

## Change Log Template

Use this template for future changes:

```markdown
### [Version] - YYYY-MM-DD

**Component**: [Component Name]  
**Type**: [Feature/Bugfix/Patch/Documentation/Testing]

**Changes**:
- Change item 1
- Change item 2
- Change item 3

**Testing Notes**:
- Test description and results
- Platforms tested: [Ubuntu 22.04, ROCm 7.1.0, RX 7900 XTX]

**Performance Impact**:
- Benchmark results if applicable

**Breaking Changes**:
- List any breaking changes

**Contributors**: @username1, @username2

**Related Issues**: #123, #456
```

---

## Versioning Scheme

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes or major architectural changes
- **MINOR**: New patches or features in a backward-compatible manner
- **PATCH**: Backward-compatible bug fixes or improvements

---

## Notes

- All dates in ISO 8601 format (YYYY-MM-DD)
- Link to GitHub issues when applicable
- Include performance metrics for optimization patches
- Document rollback procedures for critical patches
