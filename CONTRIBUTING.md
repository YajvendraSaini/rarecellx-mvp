# Contributing to RareCellX

First off, thank you for considering contributing to RareCellX! We welcome bug reports, feature requests, and code contributions from the community to help advance AI in healthcare.

## Getting Started

1. Before starting any major work, please open an issue to discuss your proposed changes or feature. This ensures someone else isn't already working on it and that it aligns with the project's roadmap.
2. Because the AI model weights and `.h5ad` datasets are intentionally not hosted on GitHub due to size and privacy, **you must request access to these files to run the project locally.** Please reach out to **founder@rarecellx.app** to request the data. Once received, follow the local setup guide in our `README.md`.

## Pull Request Process

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you changed any APIs, update the `README.md` or Swagger documentation accordingly.
4. Ensure your code lints properly:
   - **Frontend:** Run `npm run lint` inside the `frontend/` directory.
   - **Backend:** Ensure your Python code somewhat adheres to PEP8 guidelines.
5. Create a Pull Request summarizing the problem you are solving and the solution you have proposed.

## Reporting Bugs

Bugs are tracked as GitHub issues. Explain the problem and include additional details to help maintainers reproduce the problem:

* **Use a clear and descriptive title** for the issue to identify the problem.
* **Describe the exact steps** which reproduce the problem.
* **Provide specific examples** to demonstrate the steps, such as links to files or specific commands that fail.
* **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior.
* **Explain which behavior you expected to see instead and why.**

*(Note: For security-related vulnerabilities, do not open a public issue. Please refer to our `SECURITY.md` file.)*
