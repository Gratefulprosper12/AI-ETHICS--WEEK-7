from qiskit import QuantumCircuit, Aer, transpile, assemble
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()
backend = Aer.get_backend("qasm_simulator")
job = backend.run(assemble(transpile(qc, backend)))
result = job.result()
print(result.get_counts())
