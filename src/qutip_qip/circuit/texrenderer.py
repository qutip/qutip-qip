import os
import sys
import shutil
import tempfile
import warnings
import functools
import subprocess
import collections

from ..operations import Gate

__all__ = ["TeXRenderer", "CONVERTERS"]


class TeXRenderer:
    """
    Class to render the circuit in latex format.
    """

    def __init__(self, qc):

        self.qc = qc
        self.N = qc.N
        self.num_cbits = qc.num_cbits
        self.gates = qc.gates
        self.input_states = qc.input_states
        self.reverse_states = qc.reverse_states

        self._latex_template = r"""
        \documentclass[border=3pt]{standalone}
        \usepackage[braket]{qcircuit}
        \begin{document}
        \Qcircuit @C=1cm @R=1cm {
        %s}
        \end{document}
        """

        self._pdflatex = self._find_system_command(["pdflatex"])
        self._pdfcrop = self._find_system_command(["pdfcrop"])

    def _gate_label(self, gate):
        gate_label = gate.latex_str
        if gate.arg_label is not None:
            return r"%s(%s)" % (gate_label, gate.arg_label)
        return r"%s" % gate_label

    def latex_code(self):
        """
        Generate the latex code for the circuit.

        Returns
        -------
        code: str
            The latex code for the circuit.
        """

        rows = []

        ops = self.gates
        col = []
        for op in ops:
            if isinstance(op, Gate):
                gate = op
                col = []
                _swap_processing = False
                for n in range(self.N + self.num_cbits):
                    if gate.targets and n in gate.targets:
                        if len(gate.targets) > 1:
                            if gate.name == "SWAP":
                                if _swap_processing:
                                    col.append(r" \qswap \qw")
                                    continue
                                distance = abs(
                                    gate.targets[1] - gate.targets[0]
                                )
                                if self.reverse_states:
                                    distance = -distance
                                col.append(r" \qswap \qwx[%d] \qw" % distance)
                                _swap_processing = True

                            elif (
                                self.reverse_states and n == max(gate.targets)
                            ) or (
                                not self.reverse_states
                                and n == min(gate.targets)
                            ):
                                col.append(
                                    r" \multigate{%d}{%s} "
                                    % (
                                        len(gate.targets) - 1,
                                        self._gate_label(gate),
                                    )
                                )
                            else:
                                col.append(
                                    r" \ghost{%s} " % (self._gate_label(gate))
                                )

                        elif gate.name == "CNOT":
                            col.append(r" \targ ")
                        elif gate.name == "CY":
                            col.append(r" \targ ")
                        elif gate.name == "CZ":
                            col.append(r" \targ ")
                        elif gate.name == "CS":
                            col.append(r" \targ ")
                        elif gate.name == "CT":
                            col.append(r" \targ ")
                        elif gate.name == "TOFFOLI":
                            col.append(r" \targ ")
                        else:
                            col.append(r" \gate{%s} " % self._gate_label(gate))

                    elif gate.controls and n in gate.controls:
                        control_tag = (-1 if self.reverse_states else 1) * (
                            gate.targets[0] - n
                        )
                        col.append(r" \ctrl{%d} " % control_tag)

                    elif (
                        gate.classical_controls
                        and (n - self.N) in gate.classical_controls
                    ):
                        control_tag = (-1 if self.reverse_states else 1) * (
                            gate.targets[0] - n
                        )
                        col.append(r" \ctrl{%d} " % control_tag)

                    elif not gate.controls and not gate.targets:
                        # global gate
                        if (self.reverse_states and n == self.N - 1) or (
                            not self.reverse_states and n == 0
                        ):
                            col.append(
                                r" \multigate{%d}{%s} "
                                % (
                                    self.N - 1,
                                    self._gate_label(gate),
                                )
                            )
                        else:
                            col.append(
                                r" \ghost{%s} " % (self._gate_label(gate))
                            )
                    else:
                        col.append(r" \qw ")

            else:
                measurement = op
                col = []
                for n in range(self.N + self.num_cbits):
                    if n in measurement.targets:
                        col.append(r" \meter")
                    elif (n - self.N) == measurement.classical_store:
                        sgn = 1 if self.reverse_states else -1
                        store_tag = sgn * (n - measurement.targets[0])
                        col.append(r" \qw \cwx[%d] " % store_tag)
                    else:
                        col.append(r" \qw ")

            col.append(r" \qw ")
            rows.append(col)

        input_states_quantum = [
            r"\lstick{\ket{" + x + "}}" if x is not None else ""
            for x in self.input_states[: self.N]
        ]
        input_states_classical = [
            r"\lstick{" + x + "}" if x is not None else ""
            for x in self.input_states[self.N :]
        ]
        input_states = input_states_quantum + input_states_classical

        code = ""
        n_iter = (
            reversed(range(self.N + self.num_cbits))
            if self.reverse_states
            else range(self.N + self.num_cbits)
        )
        for n in n_iter:
            code += r" & %s" % input_states[n]
            for m in range(len(ops)):
                code += r" & %s" % rows[m][n]
            code += r" & \qw \\ " + "\n"

        return self._latex_template % code

    def raw_img(self, file_type="png", dpi=100):
        return self.image_from_latex(self.latex_code(), file_type, dpi)

    @classmethod
    def _run_command(self, command, *args, **kwargs):
        """
        Run a command with stdout explicitly thrown away, raising
        `RuntimeError` with the system error message
        if the command returned a non-zero exit code.
        """
        try:
            return subprocess.run(
                command,
                *args,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                **kwargs,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(e.stderr.decode(sys.stderr.encoding)) from None

    def _force_remove(self, *filenames):
        """`rm -f`: try to remove a file, ignoring errors if it doesn't exist."""
        for filename in filenames:
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass

    @classmethod
    def _test_convert_is_imagemagick(self):
        """
        Test to see if the `convert` command behaves like we'd expect ImageMagick
        to.  On Windows if ImageMagick is not installed then `convert` may refer to
        a system utility.
        """
        try:
            # Don't use `capture_output` because we're still supporting Python 3.6
            process = subprocess.run(
                ("convert", "-version"),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            return "imagemagick" in process.stdout.decode("utf-8").lower()
        except FileNotFoundError:
            return False

    @staticmethod
    def _find_system_command(names):
        """
        Given a list of possible system commands (as strings), return the first one
        which has a locatable executable form, or `None` if none of them do.  We
        also check some special cases of shadowing (e.g. ImageMagick 6's `convert`
        is also a Windows system utility) to try and catch false-positives.
        """
        for name in names:
            if shutil.which(name) is not None:
                is_valid = _SPECIAL_CASES.get(name, lambda: True)()
                if is_valid:
                    return name
        return None

    def _crop_pdf(self, filename):
        if self._pdfcrop is not None:
            """Crop the pdf file `filename` in place."""
            temporary = ".tmp." + filename
            self._run_command((self._pdfcrop, filename, temporary))
            # Windows does not allow renaming to an existing file (but unix does).
            self._force_remove(filename)
            os.rename(temporary, filename)
        else:
            # Warn, but do not raise - we can recover from a failed crop.
            warnings.warn(
                "Could not locate system 'pdfcrop':"
                " image output may have additional margins."
            )

    @staticmethod
    def _convert_pdf(file_stem, dpi=None):
        """
        'Convert' to pdf: since LaTeX outputs a PDF file, there's nothing to do.
        """
        if dpi is not None:
            warnings.warn("argument dpi is ignored for pdf output.")
        with open(file_stem + ".pdf", "rb") as file:
            return file.read()

    @classmethod
    def _make_converter(self, configuration):
        """
        Create the actual conversion function of signature
            file_stem: str -> 'T,
        where 'T is data in the format to be converted to.
        """
        which = self._find_system_command(configuration.executables)
        if which is None:
            return None
        mode = "rb" if configuration.binary else "r"

        def converter(file_stem, dpi=100):
            """
            Convert a file located in the current directory named `<file_stem>.pdf`
            to an image format with the name `<file_stem>.xxx`, where `xxx` is
            converter-dependent.

            Parameters
            ----------
            file_stem : str
                The basename of the PDF file to be converted.
            dpi : int/float
                Image density in dots per inch. Ignored for SVG.
            """
            in_file = file_stem + ".pdf"
            out_file = file_stem + "." + configuration.file_type
            if "-density" in configuration.arguments:
                arguments = list(configuration.arguments)
                arguments[arguments.index("-density") + 1] = str(dpi)
            else:
                arguments = configuration.arguments
            self._run_command((which, *arguments, in_file, out_file))
            with open(out_file, mode) as file:
                return file.read()

        return converter

    def image_from_latex(self, code, file_type="png", dpi=100):
        """
        Convert the LaTeX `code` into an image format, defined by the
        `file_type`.  Returns a string or bytes object, depending on whether
        the requested type is textual (e.g. svg) or binary (e.g. png).  The
        known file types are in keys in this module's `CONVERTERS` dictionary.

        Parameters
        ----------
        code: str
            LaTeX code representing the circuit to be converted.

        file_type: str ("png")
            The file type that the image should be returned in.


        Returns
        -------
        image: str or bytes
            An encoded version of the image.  Whether the output type is str or
            bytes depends on whether the requested image format is textual or
            binary.
        """
        if self._pdflatex is not None:
            filename = "qcirc"  # Arbitrary and internal.
            # We do all the image conversion in a temporary directory to prevent
            # leftover files if something goes wrong (or we get a
            # KeyboardInterrupt) during conversion.
            previous_dir = os.getcwd()
            with tempfile.TemporaryDirectory() as temporary_dir:
                try:
                    os.chdir(temporary_dir)
                    with open(filename + ".tex", "w") as file:
                        file.write(code)
                    try:
                        self._run_command(
                            (
                                self._pdflatex,
                                "-interaction",
                                "batchmode",
                                filename,
                            )
                        )
                    except RuntimeError as e:
                        message = (
                            "pdflatex failed."
                            " Perhaps you do not have it installed, or you are"
                            " missing the LaTeX package 'qcircuit'."
                        )
                        message += (
                            "The latex code is printed below. "
                            "Please try to compile locally using pdflatex:\n"
                            + code
                        )
                        raise RuntimeError(message) from e
                    self._crop_pdf(filename + ".pdf")
                    if file_type in _MISSING_CONVERTERS:
                        dependency = _MISSING_CONVERTERS[file_type]
                        message = "".join(
                            [
                                "Could not find system ",
                                dependency,
                                ".",
                                " Image conversion to '",
                                file_type,
                                "'",
                                " is not available.",
                            ]
                        )
                        raise RuntimeError(message)
                    if file_type not in CONVERTERS:
                        raise ValueError(
                            "".join(
                                ["Unknown output format: '", file_type, "'."]
                            )
                        )
                    out = CONVERTERS[file_type](filename, dpi)
                finally:
                    # Leave the temporary directory before it is removed (necessary
                    # on Windows, but it doesn't hurt on POSIX).
                    os.chdir(previous_dir)
            return out
        else:
            raise RuntimeError("Could not find system 'pdflatex'.")


# Record type to hold definitions of possible conversions - this is just for
# reading convenience.
_ConverterConfiguration = collections.namedtuple(
    "_ConverterConfiguration",
    ["file_type", "dependency", "executables", "arguments", "binary"],
)
CONVERTERS = {"pdf": TeXRenderer._convert_pdf}
_MISSING_CONVERTERS = {}
_CONVERTER_CONFIGURATIONS = [
    _ConverterConfiguration(
        "png",
        "ImageMagick",
        ["magick", "convert"],
        arguments=("-density", "100"),
        binary=True,
    ),
    _ConverterConfiguration(
        "svg", "pdf2svg", ["pdf2svg"], arguments=(), binary=False
    ),
]
_SPECIAL_CASES = {
    "convert": TeXRenderer._test_convert_is_imagemagick,
}
for configuration in _CONVERTER_CONFIGURATIONS:
    # Make the converter using a higher-order function, because if we defined a
    # function in the loop, it would be easy to later introduce bugs due to
    # leaky closures over loop variables.
    converter = TeXRenderer._make_converter(configuration)
    if converter:
        CONVERTERS[configuration.file_type] = converter
    else:
        _MISSING_CONVERTERS[configuration.file_type] = configuration.dependency
