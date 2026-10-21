# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Test Foundation Models memory management and timeout.

This test suite provides comprehensive coverage of:
1. Leak detection using weakref to verify objects are actually deallocated
2. Concurrent requests testing
3. Error path coverage to verify cleanup on errors
4. Long-running sessions with many requests
5. Tool callback memory management
6. Stream cleanup verification
"""

import asyncio
import gc
import weakref
import apple_fm_sdk as fm
import pytest
from tester_tools.tester_tools import SimpleCalculatorTool


@pytest.mark.asyncio
async def test_timeout_handling():
    """Demonstrate timeout handling for long-running requests."""
    print("\n=== Timeout Handling ===")

    # Get the default model
    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"No model available: {reason}")

    session = fm.LanguageModelSession(model=model)

    # Test with a very short timeout
    print("Testing with 0.1 second timeout (should timeout)...")
    with pytest.raises(asyncio.CancelledError):

        async def respond():
            return await session.respond(
                "Write a very long essay about artificial intelligence."
            )

        task = asyncio.create_task(respond())
        await asyncio.sleep(0.1)
        task.cancel()
        await task  # This will raise CancelledError
    print("✓ Correctly handled timeout")

    # Wait until the session stopped working on the previous request.
    print("Waiting for the underlying response request to finish...")
    while session.is_responding:
        await asyncio.sleep(0.5)
    assert not session.is_responding, "Session should have stopped"

    # Add additional delay to ensure native cleanup is complete
    # This addresses the race condition where is_responding becomes False
    # but the native layer hasn't fully cleaned up yet
    print("Waiting for native cleanup to complete...")
    await asyncio.sleep(0.2)

    # Test with no timeout
    print("Testing with no timeout...")
    response = await session.respond("What is 2+2?")
    print(f"✓ Response received: {response}")


@pytest.mark.asyncio
async def test_memory_management():
    """Test creating and releasing multiple objects."""
    print("\n=== Memory Management Test ===")

    # Create multiple models and sessions
    for i in range(5):
        # Get the default model
        model = fm.SystemLanguageModel()
        is_available, reason = model.is_available()
        if not is_available:
            pytest.skip(f"No model available: {reason}")

        session = fm.LanguageModelSession(model=model)  # noqa: F841 expected unused variable
        # Objects should be automatically released when going out of scope

    print("✓ Created and released multiple objects successfully")


# =============================================================================
# 1. Leak Detection Tests (using weakref)
# =============================================================================


@pytest.mark.asyncio
async def test_session_deallocation():
    """Verify that sessions are actually deallocated using weakref."""
    print("\n=== Testing Session Deallocation ===")

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"No model available: {reason}")

    # Create session and weak reference
    session = fm.LanguageModelSession(model=model)
    weak_session = weakref.ref(session)

    # Use session
    response = await session.respond("What is 2+2?")
    assert response, "Expected response from session"
    print(f"✓ Session responded: {response[:50]}...")

    # Delete session and verify cleanup
    del session
    gc.collect()
    gc.collect()  # Run twice to catch circular references

    # Give native layer time to clean up
    await asyncio.sleep(0.1)

    assert weak_session() is None, "Session not deallocated - memory leak detected!"
    print("✓ Session properly deallocated")


@pytest.mark.asyncio
async def test_model_deallocation():
    """Verify that models are actually deallocated."""
    print("\n=== Testing Model Deallocation ===")

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"No model available: {reason}")

    weak_model = weakref.ref(model)

    # Delete model and verify cleanup
    del model
    gc.collect()
    gc.collect()

    await asyncio.sleep(0.1)

    assert weak_model() is None, "Model not deallocated - memory leak detected!"
    print("✓ Model properly deallocated")


@pytest.mark.asyncio
async def test_generated_content_deallocation():
    """Verify that GeneratedContent objects are deallocated."""
    print("\n=== Testing GeneratedContent Deallocation ===")

    content = fm.GeneratedContent(content_dict={"test": "value"})
    weak_content = weakref.ref(content)

    # Use content
    value = content.value(str, for_property="test")
    assert value == "value", f"Unexpected value: {value}"
    print(f"✓ Content value: {value}")

    # Delete and verify cleanup
    del content
    gc.collect()
    gc.collect()

    assert weak_content() is None, "GeneratedContent not deallocated - memory leak!"
    print("✓ GeneratedContent properly deallocated")


@pytest.mark.asyncio
async def test_tool_deallocation():
    """Verify that Tool objects are deallocated."""
    print("\n=== Testing Tool Deallocation ===")

    tool = SimpleCalculatorTool()
    weak_tool = weakref.ref(tool)

    # Use tool
    args = fm.GeneratedContent(content_dict={"operation": "add", "a": 1.0, "b": 2.0})
    result = await tool.call(args)
    assert "3" in result, f"Unexpected result: {result}"
    print(f"✓ Tool result: {result}")

    # Delete and verify cleanup
    del tool
    del args
    gc.collect()
    gc.collect()

    assert weak_tool() is None, "Tool not deallocated - memory leak detected!"
    print("✓ Tool properly deallocated")


@pytest.mark.asyncio
async def test_multiple_sessions_deallocation():
    """Verify multiple sessions are all deallocated."""
    print("\n=== Testing Multiple Sessions Deallocation ===")

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"No model available: {reason}")

    weak_refs = []

    # Create multiple sessions
    for i in range(5):
        session = fm.LanguageModelSession(model=model)
        weak_refs.append(weakref.ref(session))
        response = await session.respond(f"Count to {i + 1}")
        assert response, f"Expected response for session {i}"
        print(f"✓ Session {i + 1} responded")
        del session

    # Force cleanup
    gc.collect()
    gc.collect()
    await asyncio.sleep(0.2)

    # Verify all sessions deallocated
    leaked_count = sum(1 for ref in weak_refs if ref() is not None)
    assert leaked_count == 0, f"{leaked_count} sessions not deallocated - memory leak!"
    print(f"✓ All {len(weak_refs)} sessions properly deallocated")


# =============================================================================
# 2. Concurrent Requests Tests
# =============================================================================


@pytest.mark.asyncio
async def test_concurrent_requests_queued():
    """Test that concurrent requests to same session are queued and processed sequentially."""
    print("\n=== Testing Concurrent Requests Queuing ===")

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"No model available: {reason}")

    session = fm.LanguageModelSession(model=model)

    # Track completion order
    completion_order = []

    async def first_request():
        response = await session.respond("What is 1+1?")
        completion_order.append("first")
        return response

    async def second_request():
        # This should wait for the first request to complete
        response = await session.respond("What is 2+2?")
        completion_order.append("second")
        return response

    # Start both requests concurrently
    task1 = asyncio.create_task(first_request())

    # Give first request time to start
    await asyncio.sleep(0.1)

    # Verify first request is active
    assert session.is_responding, "First request should be active"

    # Start second request - it should queue and wait
    task2 = asyncio.create_task(second_request())

    # Wait for both to complete
    response1, response2 = await asyncio.gather(task1, task2)

    # Verify both completed successfully
    assert response1, "First request should have response"
    assert response2, "Second request should have response"

    # Verify they completed in order (sequential, not concurrent)
    assert completion_order == ["first", "second"], (
        f"Requests should complete sequentially, got: {completion_order}"
    )

    print("✓ Concurrent requests properly queued and processed sequentially")
    print(f"✓ First response: {response1[:50]}...")
    print(f"✓ Second response: {response2[:50]}...")


@pytest.mark.asyncio
async def test_multiple_sessions_concurrent():
    """Test multiple sessions can run concurrently."""
    print("\n=== Testing Multiple Sessions Concurrent ===")

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"No model available: {reason}")

    # Create multiple sessions
    sessions = [fm.LanguageModelSession(model=model) for _ in range(3)]

    # Run requests concurrently across different sessions
    tasks = [
        session.respond(f"What is {i + 1} + {i + 1}?")
        for i, session in enumerate(sessions)
    ]

    import time

    start_time = time.time()
    responses = await asyncio.gather(*tasks)
    elapsed = time.time() - start_time

    # Verify all responses received
    assert len(responses) == 3, f"Expected 3 responses, got {len(responses)}"
    for i, response in enumerate(responses):
        assert response, f"Expected response from session {i}"
        print(f"✓ Session {i + 1} responded: {response[:50]}...")

    print(f"✓ All {len(sessions)} sessions completed concurrently in {elapsed:.2f}s")


@pytest.mark.asyncio
async def test_sequential_requests_same_session():
    """Test sequential requests to same session work correctly."""
    print("\n=== Testing Sequential Requests Same Session ===")

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"No model available: {reason}")

    session = fm.LanguageModelSession(model=model)

    # Make multiple sequential requests
    for i in range(3):
        response = await session.respond(f"What is {i + 1} times 2?")
        assert response, f"Expected response for request {i + 1}"
        print(f"✓ Request {i + 1} completed: {response[:50]}...")

    print("✓ All sequential requests completed successfully")


# =============================================================================
# 3. Error Path Coverage Tests
# =============================================================================


@pytest.mark.asyncio
async def test_error_path_cleanup_invalid_prompt():
    """Verify cleanup when invalid prompt causes error."""
    print("\n=== Testing Error Path Cleanup (Invalid Prompt) ===")

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"No model available: {reason}")

    session = fm.LanguageModelSession(model=model)

    # Try to send empty prompt (may cause error)
    try:
        await session.respond("")
    except Exception as e:
        print(f"✓ Error caught as expected: {type(e).__name__}")

    # Verify session is still usable after error
    response = await session.respond("What is 2+2?")
    assert response, "Session should be usable after error"
    print("✓ Session usable after error path")


@pytest.mark.asyncio
async def test_error_path_cleanup_cancelled_request():
    """Verify cleanup when request is cancelled."""
    print("\n=== Testing Error Path Cleanup (Cancelled Request) ===")

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"No model available: {reason}")

    session = fm.LanguageModelSession(model=model)

    # Start request and cancel it
    task = asyncio.create_task(
        session.respond("Write a very long essay about quantum physics")
    )

    await asyncio.sleep(0.1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    print("✓ Request cancelled successfully")

    # Wait for cleanup
    while session.is_responding:
        await asyncio.sleep(0.1)
    await asyncio.sleep(0.2)

    # Verify session is usable after cancellation
    response = await session.respond("What is 2+2?")
    assert response, "Session should be usable after cancellation"
    print("✓ Session usable after cancellation")


@pytest.mark.asyncio
async def test_error_path_cleanup_with_schema():
    """Verify cleanup when structured generation encounters error."""
    print("\n=== Testing Error Path Cleanup (With Schema) ===")

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"No model available: {reason}")

    session = fm.LanguageModelSession(model=model)

    # Create a simple schema
    @fm.generable()
    class SimpleResponse:
        answer: str

    # Try structured generation that might fail
    try:
        response = await session.respond(
            "Generate invalid JSON", generating=SimpleResponse
        )
        print(f"✓ Response received: {response.answer[:50]}...")
    except Exception as e:
        print(f"✓ Error handled: {type(e).__name__}")

    # Verify session is still usable
    response = await session.respond("What is 2+2?")
    assert response, "Session should be usable after schema error"
    print("✓ Session usable after schema error")


@pytest.mark.asyncio
async def test_error_path_multiple_cancellations():
    """Test multiple cancellations don't cause issues."""
    print("\n=== Testing Multiple Cancellations ===")

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"No model available: {reason}")

    session = fm.LanguageModelSession(model=model)

    # Cancel multiple requests
    for i in range(3):
        task = asyncio.create_task(
            session.respond(f"Write a long essay about topic {i + 1}")
        )

        await asyncio.sleep(0.05)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        print(f"✓ Cancellation {i + 1} handled")

        # Wait for cleanup
        while session.is_responding:
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.2)

    # Verify session is still usable
    response = await session.respond("What is 2+2?")
    assert response, "Session should be usable after multiple cancellations"
    print("✓ Session usable after multiple cancellations")


# =============================================================================
# 4. Long-Running Session Tests
# =============================================================================


@pytest.mark.asyncio
async def test_long_running_session_many_requests():
    """Test session with many sequential requests."""
    print("\n=== Testing Long-Running Session (Many Requests) ===")

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"No model available: {reason}")

    session = fm.LanguageModelSession(model=model)

    # Make many requests to same session
    num_requests = 20
    for i in range(num_requests):
        response = await session.respond(f"What is {i + 1}?")
        assert response, f"Expected response for request {i + 1}"
        if i % 5 == 0:
            print(f"✓ Completed {i + 1}/{num_requests} requests")

    print(f"✓ All {num_requests} requests completed successfully")


@pytest.mark.asyncio
async def test_long_running_session_with_tools():
    """Test long-running session with tool calls."""
    print("\n=== Testing Long-Running Session with Tools ===")

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"No model available: {reason}")

    tool = SimpleCalculatorTool()
    session = fm.LanguageModelSession(model=model, tools=[tool])

    # Make multiple requests that might use tools
    num_requests = 10
    for i in range(num_requests):
        response = await session.respond(f"What is {i + 1} plus {i + 2}?")
        assert response, f"Expected response for request {i + 1}"
        if i % 3 == 0:
            print(f"✓ Completed {i + 1}/{num_requests} requests with tools")

    print(f"✓ All {num_requests} requests with tools completed")


@pytest.mark.asyncio
async def test_long_running_session_mixed_operations():
    """Test session with mixed operations (respond, stream, schema)."""
    print("\n=== Testing Long-Running Session (Mixed Operations) ===")

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"No model available: {reason}")

    session = fm.LanguageModelSession(model=model)

    # Mix different types of operations
    operations = [
        ("respond", "What is 1+1?"),
        ("stream", "Count to 3"),
        ("respond", "What is 2+2?"),
        ("stream", "Say hello"),
        ("respond", "What is 3+3?"),
    ]

    for i, (op_type, prompt) in enumerate(operations):
        if op_type == "respond":
            response = await session.respond(prompt)
            assert response, f"Expected response for operation {i + 1}"
            print(f"✓ Operation {i + 1} (respond) completed")
        elif op_type == "stream":
            chunks = []
            async for chunk in session.stream_response(prompt):
                chunks.append(chunk)
            assert chunks, f"Expected chunks for operation {i + 1}"
            print(f"✓ Operation {i + 1} (stream) completed with {len(chunks)} chunks")

    print(f"✓ All {len(operations)} mixed operations completed")


# =============================================================================
# 5. Tool Memory Management Tests
# =============================================================================


@pytest.mark.asyncio
async def test_tool_callback_memory():
    """Test tool callback memory management."""
    print("\n=== Testing Tool Callback Memory ===")

    call_count = {"count": 0}
    weak_refs = []

    class MemoryTrackingTool(fm.Tool):
        name = "memory_tracker"
        description = "Track memory in tool callbacks"

        @property
        def arguments_schema(self) -> fm.GenerationSchema:
            @fm.generable()
            class Args:
                value: int

            return Args.generation_schema()

        async def call(self, args: fm.GeneratedContent) -> str:
            call_count["count"] += 1
            # Track the GeneratedContent object
            weak_refs.append(weakref.ref(args))
            value = args.value(int, for_property="value")
            return f"Processed: {value}"

    tool = MemoryTrackingTool()

    # Call tool multiple times
    for i in range(5):
        args = fm.GeneratedContent(content_dict={"value": i})
        result = await tool.call(args)
        assert f"Processed: {i}" in result, f"Unexpected result: {result}"
        del args

    print(f"✓ Tool called {call_count['count']} times")

    # Force cleanup
    gc.collect()
    gc.collect()
    await asyncio.sleep(0.1)

    # Check if GeneratedContent objects were deallocated
    leaked = sum(1 for ref in weak_refs if ref() is not None)
    assert leaked == 0, f"{leaked} GeneratedContent objects not deallocated!"
    print("✓ All GeneratedContent objects properly deallocated")


@pytest.mark.asyncio
async def test_tool_with_session_memory():
    """Test tool memory when used with session."""
    print("\n=== Testing Tool with Session Memory ===")

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"No model available: {reason}")

    tool = SimpleCalculatorTool()
    weak_tool = weakref.ref(tool)

    session = fm.LanguageModelSession(model=model, tools=[tool])
    weak_session = weakref.ref(session)

    # Use session with tool
    response = await session.respond("What is 5 plus 3?")
    assert response, "Expected response"
    print(f"✓ Session with tool responded: {response[:50]}...")

    # Delete and verify cleanup
    del session
    del tool
    gc.collect()
    gc.collect()
    await asyncio.sleep(0.2)

    assert weak_session() is None, "Session not deallocated!"
    assert weak_tool() is None, "Tool not deallocated!"
    print("✓ Session and tool properly deallocated")


@pytest.mark.asyncio
async def test_multiple_tools_memory():
    """Test memory management with multiple tools."""
    print("\n=== Testing Multiple Tools Memory ===")

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"No model available: {reason}")

    # Create multiple tools
    tools = [SimpleCalculatorTool() for _ in range(5)]
    weak_refs = [weakref.ref(tool) for tool in tools]

    session = fm.LanguageModelSession(model=model, tools=[*tools])

    # Use session
    response = await session.respond("What is 2+2?")
    assert response, "Expected response"
    print(f"✓ Session with {len(tools)} tools responded")

    # Delete and verify cleanup
    del session
    del tools
    gc.collect()
    gc.collect()
    await asyncio.sleep(0.2)

    leaked = sum(1 for ref in weak_refs if ref() is not None)
    assert leaked == 0, f"{leaked} tools not deallocated!"
    print(f"✓ All {len(weak_refs)} tools properly deallocated")


# =============================================================================
# 6. Stream Cleanup Tests
# =============================================================================


@pytest.mark.asyncio
async def test_stream_cleanup_normal():
    """Test stream cleanup after normal completion."""
    print("\n=== Testing Stream Cleanup (Normal) ===")

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"No model available: {reason}")

    session = fm.LanguageModelSession(model=model)

    # Stream response
    chunks = []
    async for chunk in session.stream_response("Count to 5"):
        chunks.append(chunk)

    assert chunks, "Expected chunks from stream"
    print(f"✓ Stream completed with {len(chunks)} chunks")

    # Verify session is still usable
    response = await session.respond("What is 2+2?")
    assert response, "Session should be usable after stream"
    print("✓ Session usable after stream completion")


@pytest.mark.asyncio
async def test_stream_cleanup_early_break():
    """Test stream cleanup when breaking early."""
    print("\n=== Testing Stream Cleanup (Early Break) ===")

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"No model available: {reason}")

    session = fm.LanguageModelSession(model=model)

    # Stream response but break early
    chunk_count = 0
    async for chunk in session.stream_response("Write a long story"):
        chunk_count += 1
        if chunk_count >= 3:
            break

    print(f"✓ Stream broke early after {chunk_count} chunks")

    # Wait for cleanup
    await asyncio.sleep(0.2)

    # Verify session is still usable
    response = await session.respond("What is 2+2?")
    assert response, "Session should be usable after early break"
    print("✓ Session usable after early stream break")


@pytest.mark.asyncio
async def test_stream_cleanup_exception():
    """Test stream cleanup when exception occurs."""
    print("\n=== Testing Stream Cleanup (Exception) ===")

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"No model available: {reason}")

    session = fm.LanguageModelSession(model=model)

    # Stream response and raise exception
    chunk_count = 0
    try:
        async for chunk in session.stream_response("Count to 10"):
            chunk_count += 1
            if chunk_count >= 2:
                raise ValueError("Test exception")
    except ValueError:
        print(f"✓ Exception raised after {chunk_count} chunks")

    # Wait for cleanup
    await asyncio.sleep(0.2)

    # Verify session is still usable
    response = await session.respond("What is 2+2?")
    assert response, "Session should be usable after exception"
    print("✓ Session usable after stream exception")


@pytest.mark.asyncio
async def test_multiple_streams_sequential():
    """Test multiple sequential streams."""
    print("\n=== Testing Multiple Sequential Streams ===")

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"No model available: {reason}")

    session = fm.LanguageModelSession(model=model)

    # Run multiple streams sequentially
    for i in range(3):
        chunks = []
        async for chunk in session.stream_response(f"Count to {i + 1}"):
            chunks.append(chunk)

        assert chunks, f"Expected chunks from stream {i + 1}"
        print(f"✓ Stream {i + 1} completed with {len(chunks)} chunks")

    print("✓ All sequential streams completed successfully")


@pytest.mark.asyncio
async def test_stream_session_deallocation():
    """Test session deallocation after streaming."""
    print("\n=== Testing Stream Session Deallocation ===")

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"No model available: {reason}")

    session = fm.LanguageModelSession(model=model)
    weak_session = weakref.ref(session)

    # Stream response
    chunks = []
    async for chunk in session.stream_response("Say hello"):
        chunks.append(chunk)

    assert chunks, "Expected chunks from stream"
    print(f"✓ Stream completed with {len(chunks)} chunks")

    # Delete and verify cleanup
    del session
    gc.collect()
    gc.collect()
    await asyncio.sleep(0.2)

    assert weak_session() is None, "Session not deallocated after streaming!"
    print("✓ Session properly deallocated after streaming")


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(120)  # 2 minute timeout for the entire test
async def test_comprehensive_memory_lifecycle():
    """Comprehensive test of memory lifecycle across all operations."""
    print("\n=== Testing Comprehensive Memory Lifecycle ===")

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"No model available: {reason}")

    weak_refs = {"sessions": [], "tools": [], "contents": []}

    # Create and use multiple sessions with various operations
    for i in range(3):
        print(f"Starting lifecycle iteration {i + 1}...")

        tool = SimpleCalculatorTool()
        weak_refs["tools"].append(weakref.ref(tool))

        session = fm.LanguageModelSession(model=model, tools=[tool])
        weak_refs["sessions"].append(weakref.ref(session))

        # Regular respond with timeout
        print(f"  - Sending respond request {i + 1}...")
        try:
            response = await asyncio.wait_for(
                session.respond(f"What is {i + 1}?"), timeout=30.0
            )
            assert response, f"Expected response {i + 1}"
            print(f"  - Respond {i + 1} completed")
        except asyncio.TimeoutError:
            print(f"  ✗ TIMEOUT: respond request {i + 1} hung after 30 seconds")
            raise AssertionError(f"Respond request {i + 1} timed out after 30 seconds")

        # Stream with timeout
        print(f"  - Starting stream request {i + 1}...")
        try:
            chunks = []
            stream_gen = session.stream_response(f"Count to {i + 1}")

            async def collect_chunks():
                async for chunk in stream_gen:
                    chunks.append(chunk)

            await asyncio.wait_for(collect_chunks(), timeout=30.0)
            assert chunks, f"Expected chunks {i + 1}"
            print(f"  - Stream {i + 1} completed with {len(chunks)} chunks")
        except asyncio.TimeoutError:
            print(f"  ✗ TIMEOUT: stream request {i + 1} hung after 30 seconds")
            raise AssertionError(f"Stream request {i + 1} timed out after 30 seconds")

        # Create content
        content = fm.GeneratedContent(content_dict={"value": i})
        weak_refs["contents"].append(weakref.ref(content))
        del content

        print(f"✓ Lifecycle iteration {i + 1} completed")

        del session
        del tool

    # Force cleanup
    print("Starting cleanup phase...")
    gc.collect()
    gc.collect()
    await asyncio.sleep(0.3)

    # Verify all objects deallocated
    print("Verifying deallocation...")
    leaked_sessions = sum(1 for ref in weak_refs["sessions"] if ref() is not None)
    leaked_tools = sum(1 for ref in weak_refs["tools"] if ref() is not None)
    leaked_contents = sum(1 for ref in weak_refs["contents"] if ref() is not None)

    assert leaked_sessions == 0, f"{leaked_sessions} sessions leaked!"
    assert leaked_tools == 0, f"{leaked_tools} tools leaked!"
    assert leaked_contents == 0, f"{leaked_contents} contents leaked!"

    print("✓ All objects properly deallocated in comprehensive test")
    print(f"  - Sessions: {len(weak_refs['sessions'])} created, 0 leaked")
    print(f"  - Tools: {len(weak_refs['tools'])} created, 0 leaked")
    print(f"  - Contents: {len(weak_refs['contents'])} created, 0 leaked")
