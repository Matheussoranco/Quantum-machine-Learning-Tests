import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter

def create_quantum_layer(num_qubits):
    qc = QuantumCircuit(num_qubits)
    weights = [Parameter(f'w{i}') for i in range(num_qubits)]
    
    qc.h(range(num_qubits))
    
    for i, weight in enumerate(weights):
        qc.ry(weight, i)
    
    for i in range(num_qubits - 1):
        qc.cz(i, i + 1)
    
    return qc, weights

def create_quantum_neural_network(num_neurons, num_layers):
    full_circuit = QuantumCircuit(num_neurons)
    weight_layers = []
    
    for layer in range(num_layers):
        layer_circuit, weights = create_quantum_layer(num_neurons)
        full_circuit.compose(layer_circuit, inplace=True)
        weight_layers.extend(weights)
    
    full_circuit.measure_all()
    
    return full_circuit, weight_layers

def cost_function(expected_output, measured_counts, shots):
    measured_prob = measured_counts.get(expected_output, 0) / shots
    return (1 - measured_prob)**2

def train_network(num_neurons, num_layers, training_data, epochs=50, learning_rate=0.1, shots=1024):
    circuit, parameters = create_quantum_neural_network(num_neurons, num_layers)
    simulator = Aer.get_backend('qasm_simulator')
    
    weights = {param: np.random.uniform(0, 2 * np.pi) for param in parameters}
    
    cost_history = []
    
    for epoch in range(epochs):
        total_cost = 0
        for input_data, expected_output in training_data:
            bound_circuit = circuit.bind_parameters(weights)
            result = execute(bound_circuit, simulator, shots=shots).result()
            counts = result.get_counts()
            
            cost = cost_function(expected_output, counts, shots)
            total_cost += cost
            
            for param in parameters:
                grad = (cost_function(expected_output, counts, shots) - cost) / learning_rate
                weights[param] -= learning_rate * grad
        
        avg_cost = total_cost / len(training_data)
        cost_history.append(avg_cost)
        
        print(f"Epoch {epoch + 1}/{epochs}, Cost: {avg_cost:.4f}")
    
    return weights, cost_history

training_data = [
    ('0000', '0010'),
    ('1111', '1100'),
    ('1010', '0101')
]

num_neurons = 4
num_layers = 3

trained_weights, training_history = train_network(num_neurons, num_layers, training_data)

plt.plot(training_history)
plt.title("Evolução do Custo Durante o Treinamento")
plt.xlabel("Épocas")
plt.ylabel("Custo")
plt.grid()
plt.show()
