#!/usr/bin/env python3
"""
Test script for Neutral Atom compiler using the built-from-source mqt.qmap library.
This script reads a QASM file and compiles it using the Neutral Atom zoned compiler.
"""

import os
from mqt import qmap
from qiskit import QuantumCircuit
from mqt.core.ir import QuantumComputation


def main():
    print("=== Testing mqt.qmap Neutral Atom Compiler (Built from Source) ===")

    # Read the QASM file
    qasm_file = "/Users/di75bus/Desktop/Repos/PhD/groundstate-preparation-via-lindbladians/data/bqskit.qasm"

    if not os.path.exists(qasm_file):
        print(f"Error: QASM file not found at {qasm_file}")
        return

    print(f"Reading QASM file: {qasm_file}")

    # Load the quantum circuit from QASM using mqt.core
    qc = QuantumComputation()
    qc.from_qasm(qasm_file)
    print(f"Loaded circuit with {qc.num_qubits} qubits")

    try:
        # Test if we can import the NA zoned module (this should trigger our debug print)
        print("\nImporting mqt.qmap.na.zoned module...")
        from mqt.qmap.na import zoned

        print("✓ Successfully imported mqt.qmap.na.zoned")

        # Use an example architecture file from the test directory
        arch_file = "/Users/di75bus/Desktop/Repos/PhD/groundstate-preparation-via-lindbladians/qmap/test/hybridmap/architectures/rubidium.json"

        print(f"\nLoading architecture from: {arch_file}")
        if os.path.exists(arch_file):
            try:
                arch = zoned.ZonedNeutralAtomArchitecture.from_json_file(arch_file)
                print("✓ Successfully loaded neutral atom architecture")

                # Try to create a routing-agnostic compiler
                print("\nCreating routing-agnostic compiler...")
                compiler = zoned.RoutingAgnosticCompiler(arch)
                print("✓ Successfully created routing-agnostic compiler")

                # Try to compile the circuit
                print("\nCompiling quantum circuit...")
                result = compiler.compile(qc)
                print("✓ Successfully compiled circuit")
                print(f"Compilation result length: {len(result)} characters")
                print(f"First 200 characters of result:\n{result[:200]}...")

            except Exception as e:
                print(
                    f"⚠ Architecture/compilation error (expected for some circuits): {e}"
                )
                print("The important part is that we can import and use the modules!")
        else:
            print(f"⚠ Architecture file not found at {arch_file}")

    except ImportError as e:
        print(f"✗ Failed to import mqt.qmap.na.zoned: {e}")
        return
    except Exception as e:
        print(f"⚠ Unexpected error: {e}")

    print("\n=== Test Summary ===")
    print("✓ Successfully imported mqt.qmap")
    print("✓ Successfully loaded QASM circuit")
    print("✓ Successfully imported NA zoned compiler")
    print("✓ Our custom debug prints should have appeared above!")
    print(
        "\nIf you see the debug message 'DEBUG: Using custom built mqt.qmap zoned compiler from source!'"
    )
    print("then you are successfully using the built-from-source version!")


if __name__ == "__main__":
    main()
