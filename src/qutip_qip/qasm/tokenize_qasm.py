import re
from itertools import chain

def _tokenize_line(command):
    """
    Tokenize (break into several parts a string of) a single line of QASM code.

    Parameters
    ----------
    command : str
        One line of QASM code to be broken into "tokens".

    Returns
    -------
    tokens : list of str
        The tokens (parts) corresponding to the qasm line taken as input.
    """

    # For gates without arguments
    if "(" not in command:
        tokens = list(chain(*[a.split() for a in command.split(",")]))
        tokens = [token.strip() for token in tokens]

    # For classically controlled gates
    elif re.match(r"\s*if\s*\(", command):
        groups = re.match(r"\s*if\s*\((.*)\)\s*(.*)\s+\((.*)\)(.*)", command)
        # for classically controlled gates with arguments
        if groups:
            tokens = ["if", "(", groups.group(1), ")"]
            tokens_gate = _tokenize_line(
                "{} ({}) {}".format(
                    groups.group(2), groups.group(3), groups.group(4)
                )
            )
            tokens += tokens_gate
        # for classically controlled gates without arguments
        else:
            groups = re.match(r"\s*if\s*\((.*)\)(.*)", command)
            tokens = ["if", "(", groups.group(1), ")"]
            tokens_gate = _tokenize_line(groups.group(2))
            tokens += tokens_gate
        tokens = [token.strip() for token in tokens]

    # For gates with arguments
    else:
        groups = re.match(r"(^.*?)\((.*)\)(.*)", command)
        if not groups:
            raise SyntaxError("QASM: Incorrect bracket formatting")
        tokens = groups.group(1).split()
        tokens.append("(")
        tokens += groups.group(2).split(",")
        tokens.append(")")
        tokens += groups.group(3).split(",")
        tokens = [token.strip() for token in tokens]

    return tokens


def tokenize_qasm(token_cmds):
    """
    Tokenize QASM code for processing, i.e. break it into several parts.

    Parameters
    ----------
    token_cmds : list of str
        Lines of QASM code.

    Returns
    -------
    tokens : list of (list of str)
        List of tokens corresponding to each QASM line taken as input.
    """

    processed_commands = []

    for line in token_cmds:
        # carry out some pre-processing for convenience
        for c in "[]()":
            line = line.replace(c, " " + c + " ")
        for c in "{}":
            line = line.replace(c, " ; " + c + " ; ")
        line_commands = line.split(";")
        line_commands = list(filter(lambda x: x != "", line_commands))

        for command in line_commands:
            tokens = _tokenize_line(command)
            processed_commands.append(tokens)

    return list(filter(lambda x: x != [], processed_commands))
