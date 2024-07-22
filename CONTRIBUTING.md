readme# Contributing to AIY Voice Assistant for Children

[Previous sections remain unchanged]

## Development Environment

This project uses a virtual environment to manage dependencies. Before starting development:

1. Ensure you have Python 3.7+ installed on your system.

2. Set up a virtual environment in the project directory:
   ```
   python3 -m venv venv
   ```

3. Activate the virtual environment:
   - On Unix or MacOS:
     ```
     source venv/bin/activate
     ```
   - On Windows:
     ```
     venv\Scripts\activate
     ```

4. Install the project dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Always ensure your virtual environment is activated when working on the project.

## Pull Request Process

[Update the first step of the existing Pull Request Process]

1. Fork the repo and create your branch from `main`. Ensure you're working within the virtual environment.

[Rest of the Pull Request Process and other sections remain unchanged]

## Testing

- Add tests for new features, especially those involving language processing or child interactions
- Ensure all tests pass before submitting a pull request
- For hardware-related changes, test thoroughly with the Google Voice Kit V2
- Always run tests within the project's virtual environment to ensure consistent results

[Rest of the document remains unchanged]