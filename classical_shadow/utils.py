import os

from dotenv import load_dotenv
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as NoisySampler
from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

load_dotenv()


def get_sampler(method: str = "perfect"):
    """
    Initialise un sampler selon la méthode choisie.

    Returns:
        tuple: (sampler, pass_manager)
    """
    if method == "perfect":
        return StatevectorSampler(), None

    elif method == "noisy":
        fake_backend = FakeManilaV2()
        noisy_simulator = AerSimulator.from_backend(fake_backend)

        sampler = NoisySampler.from_backend(noisy_simulator)
        pm = generate_preset_pass_manager(optimization_level=1, backend=fake_backend)
        return sampler, pm

    elif method == "real_hardware":
        token = os.getenv("IBM_API_KEY")
        if not token:
            raise ValueError(
                "IBM_API_KEY non trouvé dans les variables d'environnement (.env)"
            )

        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        backend = service.least_busy(simulator=False, operational=True)
        print(f"--- Connecté au hardware réel : {backend.name} ---")

        sampler = RuntimeSampler(mode=backend)
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        return sampler, pm

    else:
        raise ValueError(
            f"Méthode inconnue: {method}. Choisir 'perfect', 'noisy', ou 'real_hardware'."
        )
