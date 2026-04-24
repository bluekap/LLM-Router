# Contributing to LLM-Router

Thank you for your interest in contributing to LLM-Router! This document provides guidelines for contributing to this project.

## Table of Contents

- [How to Contribute](#how-to-contribute)
- [Setting Up Your Development Environment](#setting-up-your-development-environment)
- [Running Tests](#running-tests)
- [Submitting Changes](#submitting-changes)
- [Code Style](#code-style)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## How to Contribute

There are many ways to contribute to LLM-Router:

- Reporting bugs.
- Suggesting new features or enhancements.
- Writing documentation.
- Fixing bugs or implementing new features.
- Improving the existing codebase.

## Setting Up Your Development Environment

To get started with development, follow these steps:

1.  **Fork the repository:** Click the "Fork" button at the top right of the LLM-Router GitHub page.
2.  **Clone your forked repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/LLM-Router.git
    cd LLM-Router
    ```
3.  **Set up the virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
4.  **Configure API Keys (for local testing):**
    Copy the example config and fill in your actual API keys. These keys are ignored by Git.
    ```bash
    cp keys.json.example keys.json
    ```

## Running Tests

Before submitting any changes, please ensure that all tests pass. To run the tests:

```bash
pytest
```

If you add new features, please add corresponding tests. If you fix a bug, consider adding a test that reproduces the bug before your fix and passes after your fix.

## Submitting Changes

1.  **Create a new branch:**
    ```bash
    git checkout -b feature/your-feature-name-or-bugfix/your-bugfix-name
    ```
    Choose a descriptive name for your branch.

2.  **Make your changes:** Implement your feature or bug fix.

3.  **Commit your changes:** Write clear and concise commit messages.
    ```bash
    git add .
    git commit -m "feat: Add your feature" # or "fix: Fix your bug"
    ```

4.  **Push your branch to your forked repository:**
    ```bash
    git push origin feature/your-feature-name
    ```

5.  **Create a Pull Request:** Go to the original LLM-Router repository on GitHub and open a new Pull Request. Provide a clear description of your changes.

## Code Style

We generally follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style. We also use `black` for code formatting and `isort` for import sorting. Please ensure your code conforms to these standards. You can run pre-commit hooks to automatically format and lint your code:

```bash
pre-commit install
```

## Reporting Bugs

If you find a bug, please open an issue on the GitHub issue tracker. Include the following information:

- A clear and concise description of the bug.
- Steps to reproduce the behavior.
- Expected behavior.
- Screenshots or error messages if applicable.
- Your operating system and Python version.

## Suggesting Enhancements

If you have an idea for a new feature or an improvement, please open an issue on the GitHub issue tracker. Describe your suggestion clearly and explain why it would be beneficial to the project.