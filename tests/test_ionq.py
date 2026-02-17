import unittest
from unittest.mock import patch, MagicMock
from qutip_qip.circuit import QubitCircuit, Gate
from qutip_qip.ionq import (
    IonQProvider,
    IonQSimulator,
    IonQQPU,
    convert_qutip_circuit,
    convert_ionq_response_to_circuitresult,
    Job,
)


class TestConverter(unittest.TestCase):
    def test_convert_qutip_circuit(self):
        # Create a simple QubitCircuit with one gate for testing
        qc = QubitCircuit(N=1)
        qc.add_gate("H", targets=[0])
        qc.add_gate("CNOT", targets=[0], controls=[1])
        # Convert the qutip_qip circuit to IonQ format
        ionq_circuit = convert_qutip_circuit(qc)
        expected_output = [
            {"gate": "H", "target": 0},
            {"gate": "CNOT", "target": 0, "control": 1},
        ]
        self.assertEqual(ionq_circuit, expected_output)


class TestIonQBackend(unittest.TestCase):
    def setUp(self):
        self.provider = IonQProvider(token="dummy_token")

    @patch("qutip_qip.ionq.IonQProvider")
    def test_simulator_initialization(self, mock_provider):
        simulator = IonQSimulator(provider=mock_provider)
        self.assertEqual(simulator.provider, mock_provider)
        self.assertEqual(mock_provider.gateset, "qis")

    @patch("qutip_qip.ionq.IonQProvider")
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
            gateset="qis",
            headers={},
            url="http://dummy_url",
        )
        job.submit()
        self.assertEqual(job.id, "test_job_id")

    @patch("requests.get")
    def test_get_results(self, mock_get):
        # Simulate the status check response and the final result response
        mock_get.side_effect = [
            MagicMock(
                json=lambda: {"status": "completed"}
            ),  # Simulated status check response
            MagicMock(
                json=lambda: {"0": 0.5, "1": 0.5}
            ),  # Simulated final results response
        ]
        job = Job(
            circuit={},
            shots=1024,
            backend="simulator",
            gateset="qis",
            headers={},
            url="http://dummy_url",
        )
        job.id = (
            "test_job_id"  # Simulate a job that has already been submitted
        )
        results = job.get_results(polling_rate=0)
        self.assertEqual(
            results.get_final_states(),
            convert_ionq_response_to_circuitresult(
                {"0": 0.5, "1": 0.5}
            ).get_final_states(),
        )
