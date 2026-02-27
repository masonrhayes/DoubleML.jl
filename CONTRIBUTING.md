# Contributing to DoubleML.jl

Thank you for your interest in contributing to DoubleML.jl! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Julia 1.10 or higher
- Git

### Setting up the Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:

   ```bash
   git clone https://github.com/YOUR_USERNAME/DoubleML.jl.git
   cd DoubleML.jl
   ```
3. Install the package in development mode:

   ```julia
   using Pkg
   Pkg.develop(path=".")
   ```
4. Install all dependencies:

   ```bash
   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   ```

### Running Tests

To run the test suite:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Or from the Julia REPL:

```julia
using Pkg
Pkg.test("DoubleML")
```

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with the following information:

- A clear, descriptive title
- A minimal working example (MWE) that reproduces the bug
- Your Julia version (`versioninfo()`)
- Package versions (`Pkg.status()`)
- Expected behavior vs actual behavior
- Any error messages or stack traces

### Requesting Features

Feature requests are welcome! Please open an issue with:

- A clear description of the feature
- The motivation/use case
- Any references to similar implementations (e.g., in the Python DoubleML package)

### Contributing Code

1. **Create a branch** for your contribution:

   ```bash
   git checkout -b feature/my-new-feature
   ```
2. **Make your changes** following the [Code Style Guidelines](#code-style-guidelines)
3. **Add tests** for new functionality
4. **Run the test suite** to ensure nothing is broken:

   ```bash
   julia --project=. -e 'using Pkg; Pkg.test()'
   ```
5. **Commit your changes** with a clear message:

   ```bash
   git commit -m "Add feature: brief description"
   ```
6. **Push to your fork**:

   ```bash
   git push origin feature/my-new-feature
   ```
7. **Open a Pull Request** on GitHub

## Code Style Guidelines

This package follows the [SciML Style Guide](https://github.com/SciML/SciMLStyle). Key points:

### Formatting

- **All code must be formatted with [Runic.jl](https://github.com/fredrikekre/Runic.jl)**
- Run Runic before committing: `julia -e 'using Runic; Runic.main(["src", "test", "docs"])'`
- The CI will check formatting and reject PRs that aren't formatted

### Naming Conventions

- Modules: `PascalCase` (e.g., `DoubleML`)
- Types: `PascalCase` (e.g., `DoubleMLPLR`)
- Functions: `snake_case` (e.g., `fit!`, `bootstrap_se`)
- Constants: `UPPERCASE` or `PascalCase` for exported constants
- Private/internal functions: `_leading_underscore`

### Documentation

- All exported functions must have docstrings
- Use the standard Julia docstring format
- Include examples in docstrings when helpful
- Follow the style shown in existing docstrings

### Type Stability

- Prefer type-stable code
- Use parametric types when appropriate
- Avoid over-constraining types in function signatures

### Testing

- All new functionality must have tests
- Use `SafeTestsets.jl` for test isolation
- Include edge cases in tests
- Test against Python DoubleML for validation when applicable

## Pull Request Guidelines

### Before Submitting

- [ ] Code is formatted with Runic
- [ ] Tests pass locally
- [ ] New functionality has tests
- [ ] Documentation is updated (if needed)
- [ ] Docstrings are added for new exported functions

### PR Description

Please include:

- What changes were made and why
- Any related issues (e.g., "Closes #123")
- Breaking changes (if any)
- Testing performed

### Review Process

- Maintainers will review your PR
- Address any requested changes
- CI checks must pass before merging

## Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and PRs where appropriate

Areas for Contribution

We especially welcome contributions in these areas:

1. **New Models**: Additional DoubleML models from the Python package
2. **Performance**: Speed optimizations and benchmarking
3. **Documentation**: Tutorials, examples, and API documentation improvements
4. **Testing**: Additional test coverage and validation against Python implementation
5. **Integration**: Better integration with MLJ ecosystem and other Julia packages

## Questions?

- Check the [User Guide](https://masonrhayes.github.io/DoubleML.jl/stable/user-guide/)
- Open a [GitHub Discussion](https://github.com/masonrhayes/DoubleML.jl/discussions)
- Read the [API Documentation](https://masonrhayes.github.io/DoubleML.jl/stable/api/)

## License

By contributing to DoubleML.jl, you agree that your contributions will be licensed under the MIT License.
