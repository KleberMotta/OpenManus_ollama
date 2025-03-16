# OpenManus Development Guidelines

## Setup and Build Commands
- Setup: `./setup.sh` (Linux/macOS) or `setup.bat` (Windows)
- Install: `pip install -e .` or `python setup.py install`
- Run: `python main.py` or `python run_flow.py`
- Run single test: `python tests/test_html_size.py` (similarly for other test files)

## Code Style Guidelines
- **Type Hints**: Use comprehensive type annotations from `typing` module
- **Classes**: Leverage Pydantic models for data validation and serialization
- **Naming**: snake_case for methods/variables, PascalCase for classes
- **Imports**: Group standard library, third-party, and local imports
- **Error Handling**: Use try/except with specific exception types and logging
- **Logging**: Use loguru for structured logging with appropriate levels
- **Documentation**: Include docstrings for classes and important methods
- **Async/Await**: Use async patterns consistently for I/O operations
- **Dependencies**: Specify exact versions with ~= or >= operators

## Project Structure
- `/app`: Core application code and modules
- `/config`: Configuration files
- `/examples`: Example scripts and use cases
- `/tests`: Test files (run individually)