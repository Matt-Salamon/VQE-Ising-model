# Script to perform a simple VQE calculation in Qiskit Runtime

# Basic imports
import numpy as np
from functools import reduce

from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import EfficientSU2 # ansatz
from qiskit.opflow import X, Z, I # operators
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit_ibm_runtime import QiskitRuntimeService

# Global variables
NUM_QUBITS = 4
ANSATZ = EfficientSU2(NUM_QUBITS, entanglement='linear',
                      insert_barriers=True, su2_gates=['ry', 'rz'], reps=2)
OPTIMIZER = SPSA(maxiter=1000)

H_BIAS = 0.1

H_LIST = (10 ** np.linspace(-1.5, 1.5, 7)).tolist()
H_LIST.append(1.5)
H_LIST.append(1.25)
H_LIST.append(0.75)
H_LIST.append(2.0)
H_LIST.sort()


# Function to produce the Hamiltonian operator
def ising_hamiltonian(n_qubits, h, h_z=0., J=1.):
    first_term_operators = np.full(n_qubits, I)
    second_term_operators = np.full(n_qubits, I)
    third_term_operators = np.full(n_qubits, I)
    
    first_term_operators[0] = X
    second_term_operators[0] = Z
    second_term_operators[1] = Z
    third_term_operators[0] = Z
    
    first_term = reduce(lambda x, y: x.tensor(y), first_term_operators)
    second_term = reduce(lambda x, y: x.tensor(y), second_term_operators)
    third_term = reduce(lambda x, y: x.tensor(y), third_term_operators)
    
    for i in range(n_qubits-1):
        first_term_operators = np.roll(first_term_operators, 1)
        first_term += reduce(lambda x, y: x.tensor(y), first_term_operators)
        
        second_term_operators = np.roll(second_term_operators, 1)
        second_term += reduce(lambda x, y: x.tensor(y), second_term_operators)
        
        third_term_operators = np.roll(third_term_operators, 1)
        third_term += reduce(lambda x, y: x.tensor(y), third_term_operators)
        
    hamiltonian = - h * first_term - J * second_term - h_z * third_term
    
    return hamiltonian


# Function to produce magnetisation operators
def magnetisation_operator(n_qubits, axis):
    terms = np.full(n_qubits, I)
    if axis == 'x':
        terms[0] = X
    elif axis =='z':
        terms[0] = Z
    else:
        print("Invalid argument. Axis must be 'x' or 'y'!")
        return None
    
    operator = reduce(lambda x, y: x.tensor(y), terms)
    
    for i in range(n_qubits-1):
        terms = np.roll(terms, 1)
        operator += reduce(lambda x, y: x.tensor(y), terms)
    
    return operator


# Define the operators
X_MAGNETISATION_OP = magnetisation_operator(NUM_QUBITS, 'x')
Z_MAGNETISATION_OP = magnetisation_operator(NUM_QUBITS, 'z')

# Find the provider and backend
PROVIDER = IBMQ.load_account()
five_q_devices = PROVIDER.backends(simulator=False, operational=True,
                  filters=lambda x: x.configuration().n_qubits == 5)
BACKEND = least_busy(five_q_devices)

SERVICE = QiskitRuntimeService(channel='ibm_quantum')

for h_field in H_LIST:
    
    hamiltonian = ising_hamiltonian(NUM_QUBITS, h_field, h_z=H_BIAS)

    # Define options and inputs for Qiskit Runtime
    options = {'backend_name': BACKEND.name()}

    runtime_inputs = {
        'ansatz': ANSATZ, # object (required)
        'initial_parameters': None, # [array,string] (required)
        'operator': hamiltonian, # object (required)
        'optimizer': OPTIMIZER, # object (required)
        'aux_operators': {'x_mag': X_MAGNETISATION_OP, 'z_mag': Z_MAGNETISATION_OP}, # array
        # 'initial_layout': None, # [null,array,object]
        # 'max_evals_grouped': None, # integer
        'measurement_error_mitigation': True, # boolean
        'shots': 1024 # integer
    }

    job = SERVICE.run(program_id='vqe', options=options,
        inputs=runtime_inputs, instance='ibm-q/open/main')

    print("\n----------------------")
    print("h = {}".format(h_field))
    print(job.job_id)
    print(job.status())
