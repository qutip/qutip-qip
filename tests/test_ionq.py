import unittest
from unittest.mock import patch, MagicMock
from qutip.qip.circuit import QubitCircuit, Gate
from qutip.ionq import (
    Provider,
    IonQSimulator,
    IonQQPU,
    convert_qutip_circuit,
    Job,
)


class TestConverter(unittest.TestCase):
    def test_convert_qutip_circuit(self):
        # Create a simple QubitCircuit with one gate for testing
        qc = QubitCircuit(N=1)
        qc.add_gate("X", targets=[0])
        # Convert the qutip_qip circuit to IonQ format
        ionq_circuit = convert_qutip_circuit(qc)
        expected_output = [{"gate": "X", "targets": [0], "controls": []}]
        self.assertEqual(ionq_circuit, expected_output)


class TestIonQBackend(unittest.TestCase):
    def setUp(self):
        self.provider = Provider(token="dummy_token")

    @patch("qutip.ionq.Provider")
    def test_simulator_initialization(self, mock_provider):
        simulator = IonQSimulator(provider=mock_provider)
        self.assertEqual(simulator.provider, mock_provider)
        self.assertEqual(simulator.gateset, "qis")

    @patch("qutip.ionq.Provider")
    def test_qpu_initialization(self, mock_provider):
        qpu = IonQQPU(provider=mock_provider, qpu="harmony")
        self.assertEqual(qpu.provider, mock_provider)
        self.assertTrue("qpu.harmony" in qpu.provider.backend)


class TestJob(unittest.TestCase):
    @patch("requests.post")
    def test_submit(self, mock_post):
        mock_post.return_value.json.return_value = {"id": "test_job_id"}
        mock_post.return_value.status_code = 200
        job = Job(
            circuit={},
            shots=1024,
            backend="simulator",
            headers={},
            url="http://dummy_url",
        )
        job.submit()
        self.assertEqual(job.id, "test_job_id")

    @patch("requests.get")
    def test_get_results(self, mock_get):
        mock_get.return_value.json.return_value = {"0": 0.5, "1": 0.5}
        mock_get.return_value.status_code = 200
        job = Job(
            circuit={},
            shots=1024,
            backend="simulator",
            headers={},
            url="http://dummy_url",
        )
        job.id = (
            "test_job_id"  # Simulate a job that has already been submitted
        )
        results = job.get_results(polling_rate=0)
        self.assertEqual(results, {"0": 512, "1": 512})
