""" Tests for qutip_qip.family. """

import re

from qutip_qip import family


class TestVersion:
    def test_version(self):
        pkg, version = family.version()
        assert pkg == "qutip-qip"
        assert re.match(r"\d+\.\d+\.\d+.*", version)
