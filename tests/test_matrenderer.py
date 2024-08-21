import pytest
from unittest.mock import patch
from tests.test_textrenderer import qc1, qc2, qc3


@pytest.mark.parametrize("qc_fixture", ["qc1", "qc2", "qc3"])
def test_matrenderer(request, qc_fixture):
    qc = request.getfixturevalue(qc_fixture)

    with patch("matplotlib.pyplot.show"):
        try:
            qc.draw("matplotlib")
        except Exception as e:
            assert False, f"Error: {e}"
