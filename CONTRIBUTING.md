# Contributing to QuTiP-qip Development

First off, thanks for taking the time to contribute! ❤️

All types of contributions are encouraged and valued. Please make sure to [read the relevant section](#table-of-contents) before making your contribution.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [I Have a Question](#i-have-a-question)
- [Reporting Bugs](#reporting-bugsissues)
- [Contributing to Code](#contributing-to-code)
- [Improving the Documentation](#improving-the-documentation)
- [AI Tools Usage Policy](#ai-tools-usage-policy)


## Code of Conduct
This project and everyone participating in it are governed by the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.


## I Have a Question
You can ask questions or answer other users' questions on our [Discussion](https://github.com/qutip/qutip-qip/discussions) forum on GitHub or in the [QuTiP Google discussion group](https://groups.google.com/forum/#!forum/qutip). You may also suggest new features or improvements you would like to see in the project. We encourage you to engage with the community and share your ideas and knowledge.


## Reporting Bugs/Issues
If you find a bug or issue, it is best to first search in the existing [Issues](https://github.com/qutip/qutip-qip/issues) list. If you didn't find a related issue, please report the bug by opening a new issue:

- Create a new [Issue](https://github.com/qutip/qutip-qip/issues/new) on GitHub.
- Provide as much context as you can about what you're running into.
- Provide the python and package versions by running `qutip.about()` command.

We will then take care of the issue as soon as possible.


## Contributing to Code
If you want to contribute, please read the [Contributing to QuTiP-qip development](https://qutip-qip.readthedocs.io/en/latest/contribution-code.html) section of the documentation to set up the Python environment for development.

### Choose an issue to work on
QuTiP-qip uses the following labels to help non-maintainers find issues best suited to their interests and experience level:

- [good first issue](https://github.com/qutip/qutip-qip/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) - these issues are typically the simplest available to work on, ideal for newcomers. They should already be fully scoped, with a clear approach outlined in the descriptions.
- [help wanted](https://github.com/qutip/qutip-qip/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) - these issues are generally more complex than good first issues. They typically cover work that core maintainers don't currently have capacity to implement and may require more investigation/discussion. These are a great option for experienced contributors looking for something a bit more challenging.

### Assigning Issues
In our workflow, we generally avoid formally assigning issues to contributors. This approach is intentional — it keeps the project open and flexible, allowing anyone from the community to explore and work on problems that interest them, without waiting for explicit assignment or approval.

The only exception to this is a small subset of “core” issues. These are typically more complex, sensitive, or tightly coupled to ongoing development efforts, and are therefore handled directly by maintainers.


## Improving The Documentation
Documentation is a crucial part of any project, and we welcome contributions to improve it. If you find any errors, inconsistencies, or areas that could be clarified, please feel free to submit a pull request with your suggested changes. You can also open an issue to discuss potential improvements before making changes.

Please read the [Contributing to the documentation](https://qutip-qip.readthedocs.io/en/latest/contribution-docs.html) for more details on how to contribute to the documentation.

Also you can add new tutorials in the [Tutorials](http://github.com/qutip/qutip-tutorials/) repository. If you have an idea for a tutorial example that you think would be helpful to others, please feel free to submit a pull request with your suggested tutorial.


## AI Tools Usage Policy
We acknowledge the use of AI tools to improve efficiency and enhance quality of work. Contributors may use such tools, provided they adhere to the following guidelines:

### 1. Accountability

The human contributor is solely responsible for their contribution i.e. all the AI-generated outputs can be considered their own work. If you're submitting a Pull Request that includes AI-generated code or documentation:

- You are responsible for ensuring that code you submit meets the [project's standards](https://qutip-qip.readthedocs.io/en/latest/contribution-code.html#code-style).
- You must fully understand every line of code in the submission.
- You must be able to explain the "why" behind the implementation during the review process.

### 2. Transparency

All Pull Requests must fill the **AI Tools Usage Disclosure** in the PR template. This disclosure is mandatory and must accurately reflect how AI Tools were used.

### 3. Copyright & Legal

By submitting a contribution to qutip-qip, you agree to

1. Submit your contribution under the project's license.

2. The contribution does not violate any third-party rights or the terms of service of the AI provider. And does not include "regurgitated" code from libraries with incompatible licenses (e.g., GPL-licensed code) being suggested into our BSD-3-clause licensed project.

3. AI agents must not sign commits or be added to commit message trailer `Co-authored-by:` since copyright is fundamentally tied to the concept of human authorship as per the US Copyright law. You can instead use `Assisted-by: AI Model/Tool` as commit message trailer e.g. `Assisted-by: Cursor with Opus 4.6`.

### 4. Good Use Cases

- **Understand the codebase:** AI can be used to explore unfamiliar parts of the repository, summarize modules, and clarify how components interact. It helps build the mental model of the project faster. 

- **Brainstorming and Design:**  AI may assist in generating ideas, evaluating different approaches, and structuring designs. However, all final decisions regarding design and architecture should be independently validated and be made by the contributor.

- **Reviewing your implementation:**  AI can be used to spot potential bugs, suggest improvements, and help clarify unexpected behavior in code. Contributors should verify the correctness and relevance of AI-generated feedback before applying it.

### 5. Prohibited Use

The following use cases are prohibited:

- **Communication:** In project communications (GitHub Issues, Discussion, PR descriptions and review comments), we personally expect to communicate directly with other humans not with automated systems. Use of translation tools is completely welcome.

- **Ban on Bots/Agents:** Fully autonomous or unsupervised AI agents (e.g. OpenClaw, SWE-agent) are not allowed to submit Pull Requests.

- **Stance on Good First Issues:** The main purpose of a “good first issue” is to allow new human contributors to get familiar with `qutip-qip`, with the GitHub workflow and to engage with the community. Submitting a fully AI-generated PR defeats that purpose. AI can be used as a learning aid to understand the codebase, but the final implementation must be your own.

### 6. Enforcement

Maintainers reserve the right to close any Pull Request that violates this Policy. Maintainers may restrict participation or report users to GitHub for repeated violations of this policy.
