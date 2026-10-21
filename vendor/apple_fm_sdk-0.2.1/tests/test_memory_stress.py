# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Memory Stress Test for Foundation Models

This test performs stress testing on the Foundation Models library to detect potential
memory leaks and ensure proper resource cleanup. It creates and destroys 1000 language
model sessions in sequence, monitoring memory usage throughout the process.

Key aspects tested:
- Memory allocation and deallocation for LanguageModelSession objects
- Proper cleanup of native resources when sessions are destroyed
- Memory stability over many session creation/destruction cycles
- Garbage collection effectiveness with periodic forced collections

The test tracks Resident Set Size (RSS) memory usage and prints periodic updates.
A successful test indicates that memory usage remains stable and doesn't grow
unboundedly, confirming that sessions are properly cleaned up and no memory leaks exist.

Note: This is a standalone script (not a pytest test) that should be run directly
to observe memory behavior over time.
"""

import asyncio
import sys
import os
import apple_fm_sdk as fm
import psutil
import time
import gc

proc = psutil.Process(os.getpid())

# Configuration
NUM_ITERATIONS = 1000
GC_INTERVAL = 10
MEMORY_LEAK_THRESHOLD_MB = 50  # Maximum acceptable memory growth in MB
PAUSE_BETWEEN_REQUESTS = 0.1  # seconds


def get_memory_mb():
    """Get current RSS memory usage in MB."""
    rss = proc.memory_info().rss  # bytes
    return rss / (1024 * 1024)


def print_mem(tag):
    """Print current memory usage with a tag."""
    print(f"{tag}: RSS = {get_memory_mb():.2f} MB")


# Do not use pytest for this file
async def memory_stress():
    """
    Run memory stress test by creating many sessions.

    Returns:
        tuple: (success: bool, initial_memory_mb: float, final_memory_mb: float, error_msg: str)
    """
    print_mem("Start")

    # Force garbage collection and get baseline memory
    gc.collect()
    time.sleep(0.2)
    initial_memory = get_memory_mb()
    print(f"Baseline memory after GC: {initial_memory:.2f} MB")
    print(f"Running {NUM_ITERATIONS} iterations...\n")

    error_count = 0
    last_error = None

    for i in range(1, NUM_ITERATIONS + 1):
        try:
            model = fm.SystemLanguageModel(
                fm.SystemLanguageModelUseCase.GENERAL,
                fm.SystemLanguageModelGuardrails.DEFAULT,
            )
            session = fm.LanguageModelSession(model=model)

            # Perform a simple query
            await asyncio.sleep(PAUSE_BETWEEN_REQUESTS)
            response = await session.respond("What is 2+2?")  # noqa: F841 expected unused variable

        except fm.FoundationModelsError as e:
            error_count += 1
            last_error = str(e)
            print(f"Error at iteration {i}: {e}")
            if error_count > 10:
                return (
                    False,
                    initial_memory,
                    get_memory_mb(),
                    f"Too many errors (>10). Last error: {last_error}",
                )

        # Periodic garbage collection and memory reporting
        if i % GC_INTERVAL == 0:
            gc.collect()
            time.sleep(0.1)
            print_mem(f"Iteration {i}")

    # Final garbage collection and memory check
    gc.collect()
    time.sleep(0.2)
    final_memory = get_memory_mb()

    print(f"\n{'=' * 50}")
    print_mem("Final")
    memory_growth = final_memory - initial_memory
    print(f"Memory growth: {memory_growth:+.2f} MB")

    if error_count > 0:
        print(f"Warning: {error_count} errors occurred during test")

    # Check for memory leak
    if memory_growth > MEMORY_LEAK_THRESHOLD_MB:
        return (
            False,
            initial_memory,
            final_memory,
            f"Memory leak detected: {memory_growth:.2f} MB growth exceeds threshold of {MEMORY_LEAK_THRESHOLD_MB} MB",
        )

    return True, initial_memory, final_memory, None


async def main():
    print("Foundation Models SDK for Python - Memory stress tests")
    print("=" * 50)

    success, initial_mem, final_mem, error_msg = await memory_stress()

    print(f"\n{'=' * 50}")
    if success:
        print("✓ PASS: Memory stress test completed successfully")
        print(f"  Initial memory: {initial_mem:.2f} MB")
        print(f"  Final memory: {final_mem:.2f} MB")
        print(f"  Growth: {final_mem - initial_mem:+.2f} MB")
        return 0
    else:
        print("✗ FAIL: Memory stress test failed")
        if error_msg:
            print(f"  Reason: {error_msg}")
        print(f"  Initial memory: {initial_mem:.2f} MB")
        print(f"  Final memory: {final_mem:.2f} MB")
        print(f"  Growth: {final_mem - initial_mem:+.2f} MB")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
