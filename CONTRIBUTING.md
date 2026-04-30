# Contributing to qutip-qip

First off, thanks for taking the time to contribute! ❤️

All types of contributions are encouraged and valued. Please make sure to read the relevant sections before making your contribution.

## Table of Contents

- [Contributing to Code](#contributing-to-code)
    - [Before you start](#before-you-start)
    - [Choose an issue to work on](#choose-an-issue-to-work-on)
    - [Assigning Issues](#assigning-issues)
    - [Development Workflow](#development-workflow)
    - [Code Review](#code-review)
- [Contributing to Documentation](#contributing-to-documentation)
    - [Getting started](#getting-started)
    - [How to Contribute](#how-to-contribute)
    - [Tutorials](#tutorials)
- [AI Tools Usage Policy](#ai-tools-usage-policy)

<br>

## Contributing to Code

### Before you start
It is best to familiarize yourself with the project. You can start by reading the [documentation](https://qutip-qip.readthedocs.io/en/latest/) and trying out the examples provided in the [tutorials](https://qutip-qip.readthedocs.io/en/latest/tutorials_v5.html).

### Choose an issue to work on
QuTiP-QIP uses the following labels to help non-maintainers find issues best suited to their interests and experience level:

- [good first issue](https://github.com/qutip/qutip-qip/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) - these issues are typically the simplest available to work on, ideal for newcomers. They should typically be fully scoped, with a clear approach outlined in the descriptions.
- [help wanted](https://github.com/qutip/qutip-qip/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) - these issues are generally more complex than good first issues. They typically cover work that core maintainers don't currently have capacity to implement and may require more investigation/discussion. These are a great option for experienced contributors looking for something a bit more challenging.

### Assigning Issues
In our workflow, we generally avoid formally assigning issues to contributors. This approach is intentional — it keeps the project open and flexible, allowing anyone from the community to explore and work on problems that interest them, without waiting for explicit assignment or approval.

The only exception to this is a small subset of “core” issues. These are typically more complex, sensitive, or tightly coupled to ongoing development efforts, and are therefore handled directly by maintainers or core contributors.

### Development Workflow
Please refer to the [Contributing to Code](https://qutip-qip.readthedocs.io/en/latest/contribution-code.html) section of the documentation for details on our development workflow, including how to set up the Python environment, style code, run tests and submit Pull Requests (PR).

You may also open a draft PR with changes in order to discuss and receive feedback on the best approach if you are not sure what the best way forward is.

### Code Review
Code review is conducted openly and is accessible to everyone. While only maintainers have access to merge commits, community feedback on Pull Requests is extremely valuable. It is also a good mechanism to learn about the code base both for contributors and reviewers.

Response times may vary for your PR due to other commitments the maintainers have. If you have been waiting over a week for a review on your PR, feel free to tag a maintainer in a comment to gently remind them to review your work.

<br>

## Contributing to Documentation
Accurate and well-structured documentation is essential for making QuTiP-QIP accessible and easy to use. We welcome contributions of all sizes — from small typo fixes to entirely new sections or tutorials.

### Getting started
Before contributing, please read the [documentation guide](https://qutip-qip.readthedocs.io/en/latest/contribution-docs.html). It explains how to build the docs locally and outlines formatting and style conventions.

### How to Contribute

- **Minor fixes:** For small changes such as typos, broken links, or minor corrections, you can open a Pull Request directly.

- **Larger improvements or additions:** If your contribution involves significant changes — such as rewriting sections, adding new content, or restructuring existing material. Please open an issue first to outline the changes, once discussed you can submit a Pull Request. 

### Tutorials
Tutorials are maintained in a separate repository: [qutip-tutorials](http://github.com/qutip/qutip-tutorials/). If you wish to add a new tutorial or update an existing one, open a Pull Request directly in the qutip-tutorials repository. If you're unsure about the changes, you may open a draft PR in order to discuss and receive feedback.

<br>

## AI Tools Usage Policy
We acknowledge the use of AI tools to improve efficiency and enhance quality of work. Contributors may use such tools, provided they adhere to the following guidelines:

### 1. Accountability

The human contributor is solely responsible for their contribution i.e. all the AI-generated outputs can be considered their own work. If you're submitting a Pull Request that includes AI-generated code or documentation:

- You are responsible for ensuring that code you submit meets the project's standards.
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
