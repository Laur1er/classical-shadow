import os
import numpy as np

from dotenv import load_dotenv

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

load_dotenv()


def get_sampler(method: str = "perfect"):
    """
    Initialise a sampler according to the chosen method.

    Returns:
        tuple: (sampler, pass_manager)
    """
    if method == "perfect":
        backend = AerSimulator()
        sampler = AerSampler()

        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        return sampler, pm

    elif method == "noisy":
        fake_backend = FakeSherbrooke()
        noisy_simulator = AerSimulator.from_backend(fake_backend)

        sampler = AerSampler.from_backend(noisy_simulator)
        pm = generate_preset_pass_manager(optimization_level=1, backend=fake_backend)
        return sampler, pm

    elif method == "real_hardware":
        token = os.getenv("IBM_API_KEY")
        if not token:
            raise ValueError(
                "IBM_API_KEY not found in environnements plateforms (.env)"
            )

        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        backend = service.least_busy(simulator=False, operational=True)
        print(f"--- Connected to the QPU : {backend.name} ---")

        sampler = RuntimeSampler(mode=backend)
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        return sampler, pm

    else:
        raise ValueError(
            f"Unknown method: {method}. Choose 'perfect', 'noisy', or 'real_hardware'."
        )


def classicaly_compute_estimation_value(
    observable: SparsePauliOp, state_vector: QuantumCircuit
) -> np.complex128:
    """
    Computes classicaly the estimation value by sandwitching the observable between the statevector.
    """
    statevector = Statevector(state_vector).data
    matrix_observable = observable.to_matrix()
    real_expect_value = np.einsum(
        "i,ij,j", statevector.conj(), matrix_observable, statevector
    )
    return real_expect_value
