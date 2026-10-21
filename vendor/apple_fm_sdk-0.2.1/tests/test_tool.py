# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Test Foundation Models Tool calling functionality.

This test suite thoroughly tests the Tool class and tool calling capabilities,
including:
- Tool creation and initialization
- Tool parameter schemas
- Async callable execution
- Tool integration with LanguageModelSession
- Error handling
- Multiple tools
- Complex parameter types
"""

import asyncio
import json
import apple_fm_sdk as fm
import pytest
from tester_tools.tester_tools import (
    SimpleCalculatorTool,
    GetUserInfoTool,
    ProcessListTool,
    ErrorRaisingTool,
    AsyncDelayTool,
    CalculatorParams,
    UserInfoParams,
)


# =============================================================================
# Test Cases
# =============================================================================


@pytest.mark.asyncio
async def test_tool_creation():
    """Test basic tool creation and initialization."""
    print("\n=== Testing Tool Creation ===")

    # Create a simple tool
    tool = SimpleCalculatorTool()

    # Verify tool attributes
    assert tool.name == "simple_calculator", f"Tool name mismatch: {tool.name}"
    assert tool.description == "Perform basic arithmetic operations", (
        f"Tool description mismatch: {tool.description}"
    )
    assert isinstance(tool.arguments_schema, fm.GenerationSchema), (
        f"Invalid schema type: {type(tool.arguments_schema)}"
    )

    print(f"✓ Tool created successfully: {tool.name}")
    print(f"✓ Tool description: {tool.description}")
    print("✓ Tool has valid schema")


@pytest.mark.asyncio
async def test_tool_direct_invocation():
    """Test calling tool functions directly."""
    print("\n=== Testing Direct Tool Invocation ===")

    # Test calculator tool
    calc_tool = SimpleCalculatorTool()
    calc_args = fm.GeneratedContent(
        content_dict={"operation": "add", "a": 5.0, "b": 3.0}
    )
    result = await calc_tool.call(calc_args)
    assert "8" in result, f"Unexpected calculator result: {result}"
    print(f"✓ Calculator tool: {result}")

    # Test user info tool
    user_tool = GetUserInfoTool()
    user_args = fm.GeneratedContent(content_dict={"user_id": 1})
    result = await user_tool.call(user_args)
    assert "Alice" in result, f"Unexpected user info result: {result}"
    print(f"✓ User info tool: {result}")

    # Test list processing tool
    list_tool = ProcessListTool()
    list_args = fm.GeneratedContent(
        content_dict={"items": [1, 2, 3, 4, 5], "action": "sum"}
    )
    result = await list_tool.call(list_args)
    assert "15" in result, f"Unexpected list processing result: {result}"
    print(f"✓ List processing tool: {result}")


@pytest.mark.asyncio
async def test_tool_with_session():
    """Test tool integration with LanguageModelSession."""
    print("\n=== Testing Tool with Session ===")

    # Create tools
    calculator_tool = SimpleCalculatorTool()
    user_info_tool = GetUserInfoTool()

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"No model available: {reason}")

    # Create session with tools
    session = fm.LanguageModelSession(
        instructions="You are a helpful assistant with access to tools.",
        model=model,
        tools=[calculator_tool, user_info_tool],
    )

    print(f"✓ Created session with {len([calculator_tool, user_info_tool])} tools")

    result = await session.respond("What is 10 plus 5?")
    print(f"✓ Session response: {result}")

    result = await session.respond("Get me info about user with ID 1.")
    print(f"✓ Session response: {result}")

    print(f"✓ Session is_responding: {session.is_responding}")


@pytest.mark.asyncio
async def test_tool_error_handling(model):
    """Test tool error handling."""
    print("\n=== Testing Tool Error Handling ===")

    # Test tool that raises an error
    error_tool = ErrorRaisingTool()
    error_args = fm.GeneratedContent(content_dict={"should_fail": True})
    with pytest.raises(ValueError, match="Intentional error for testing"):
        await error_tool.call(error_args)
    print("✓ Tool correctly raised error")

    # Test tool with successful execution
    success_args = fm.GeneratedContent(content_dict={"should_fail": False})
    result = await error_tool.call(success_args)
    assert "Success" in result, f"Unexpected result: {result}"
    print(f"✓ Tool executed successfully: {result}")

    # Test division by zero
    calc_tool = SimpleCalculatorTool()
    div_zero_args = fm.GeneratedContent(
        content_dict={"operation": "divide", "a": 10.0, "b": 0.0}
    )
    result = await calc_tool.call(div_zero_args)
    assert "Error" in result, f"Expected error message: {result}"
    print(f"✓ Division by zero handled: {result}")

    # Test session with error-raising tool
    session = fm.LanguageModelSession(
        instructions="You are a helpful assistant with access to tools.",
        tools=[error_tool],
    )

    response = await session.respond(
        "You MUST execute the error tool please. Let me know what error type it throws."
    )
    assert "error" in response, f"Expected error mentioned in response: {response}"
    print(f"✓ Tool executed successfully: {response}")


@pytest.mark.asyncio
async def test_tool_with_complex_types():
    """Test tools with complex parameter types."""
    print("\n=== Testing Tools with Complex Types ===")

    # Test with list parameter
    list_tool = ProcessListTool()
    list_args = fm.GeneratedContent(
        content_dict={"items": ["apple", "banana", "cherry"], "action": "join"}
    )
    result = await list_tool.call(list_args)
    assert "apple" in result and "banana" in result, f"Unexpected result: {result}"
    print(f"✓ List parameter handled: {result}")

    # Test with numeric list
    numeric_args = fm.GeneratedContent(
        content_dict={"items": [10, 20, 30], "action": "count"}
    )
    result = await list_tool.call(numeric_args)
    assert "3" in result, f"Unexpected count result: {result}"
    print(f"✓ Numeric list handled: {result}")

    # Test with dictionary return (JSON)
    user_tool = GetUserInfoTool()
    user_args = fm.GeneratedContent(content_dict={"user_id": 2})
    result = await user_tool.call(user_args)
    user_data = json.loads(result)
    assert user_data["name"] == "Bob", f"Unexpected user data: {user_data}"
    print(f"✓ JSON return handled: {user_data}")


@pytest.mark.asyncio
async def test_tool_async_behavior():
    """Test tool async behavior and delays."""
    print("\n=== Testing Tool Async Behavior ===")

    # Test async delay tool
    delay_tool = AsyncDelayTool()
    delay_args = fm.GeneratedContent(
        content_dict={"delay": 0.1, "message": "Test message"}
    )

    import time

    start_time = time.time()
    result = await delay_tool.call(delay_args)
    elapsed = time.time() - start_time

    assert elapsed >= 0.1, f"Delay too short: {elapsed}s"
    assert "Test message" in result, f"Unexpected result: {result}"
    print(f"✓ Async delay tool executed in {elapsed:.2f}s: {result}")

    # Test concurrent tool execution
    tasks = [
        delay_tool.call(
            fm.GeneratedContent(content_dict={"delay": 0.1, "message": f"Message {i}"})
        )
        for i in range(3)
    ]

    start_time = time.time()
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start_time

    # Concurrent execution should be faster than sequential
    assert elapsed < 0.3, f"Concurrent execution too slow: {elapsed}s"
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    print(f"✓ Concurrent execution completed in {elapsed:.2f}s")


@pytest.mark.asyncio
async def test_tool_parameter_validation():
    """Test tool parameter schema validation."""
    print("\n=== Testing Tool Parameter Validation ===")

    # Create tool with constrained parameters
    tool = SimpleCalculatorTool()

    # Verify schema has correct structure
    schema_dict = tool.arguments_schema.to_dict()
    assert "properties" in schema_dict, "Schema missing properties"
    assert "operation" in schema_dict["properties"], "Schema missing operation property"
    assert "a" in schema_dict["properties"], "Schema missing a property"
    assert "b" in schema_dict["properties"], "Schema missing b property"

    print("✓ Schema structure validated")

    # Check that operation has choices constraint
    operation_prop = schema_dict["properties"]["operation"]
    if "enum" in operation_prop:
        assert "add" in operation_prop["enum"], "Missing 'add' in choices"
        assert "subtract" in operation_prop["enum"], "Missing 'subtract' in choices"
        print(f"✓ Operation choices validated: {operation_prop['enum']}")


@pytest.mark.asyncio
async def test_tool_with_all_operations():
    """Test calculator tool with all operations."""
    print("\n=== Testing All Calculator Operations ===")

    calc_tool = SimpleCalculatorTool()
    operations = [
        ("add", 10.0, 5.0, 15.0),
        ("subtract", 10.0, 5.0, 5.0),
        ("multiply", 10.0, 5.0, 50.0),
        ("divide", 10.0, 5.0, 2.0),
    ]

    for operation, a, b, expected in operations:
        args = fm.GeneratedContent(
            content_dict={"operation": operation, "a": a, "b": b}
        )
        result = await calc_tool.call(args)
        assert str(expected) in result, f"Expected {expected} in result, got: {result}"
        print(f"✓ {operation}: {a} {operation} {b} = {expected}")


@pytest.mark.asyncio
async def test_tool_schema_from_generable():
    """Test creating tool schemas from generable classes."""
    print("\n=== Testing Schema from Generable ===")

    # Create schema from generable class
    assert isinstance(CalculatorParams, fm.Generable), (
        "CalculatorParams is not Generable"
    )
    schema = CalculatorParams.generation_schema()

    assert isinstance(schema, fm.GenerationSchema), (
        f"Invalid schema type: {type(schema)}"
    )
    print("✓ Schema created from generable class")

    # Verify schema can be used in tool
    tool = SimpleCalculatorTool()

    assert isinstance(tool.arguments_schema, fm.GenerationSchema), (
        "Schema type mismatch in tool"
    )
    print("✓ Schema successfully used in tool")

    # Test schema conversion to dict
    schema_dict = schema.to_dict()
    assert "title" in schema_dict, "Schema dict missing title"
    assert "properties" in schema_dict, "Schema dict missing properties"
    print(f"✓ Schema converts to dict: {schema_dict['title']}")


@pytest.mark.asyncio
async def test_tool_lifecycle():
    """Test tool lifecycle and cleanup."""
    print("\n=== Testing Tool Lifecycle ===")

    # Create tool
    tool = SimpleCalculatorTool()

    print(f"✓ Tool created: {tool.name}")

    # Use tool
    args = fm.GeneratedContent(content_dict={"operation": "add", "a": 1.0, "b": 2.0})
    result = await tool.call(args)
    assert "3" in result, f"Unexpected result: {result}"
    print(f"✓ Tool executed: {result}")

    # Tool should still be usable after execution
    result2 = await tool.call(args)
    assert "3" in result2, f"Unexpected result on second call: {result2}"
    print("✓ Tool reusable after execution")


@pytest.mark.asyncio
async def test_tool_missing_name():
    """Test that Tool subclass without name attribute fails."""
    print("\n=== Testing Tool Missing Name ===")

    # Define a tool without name attribute
    class ToolWithoutName(fm.Tool):
        description = "A tool without a name"

        @property
        def arguments_schema(self) -> fm.GenerationSchema:
            return CalculatorParams.generation_schema()

        async def call(self, args: fm.GeneratedContent) -> str:
            return "result"

    # Try to instantiate - should fail
    with pytest.raises(AssertionError):
        tool = ToolWithoutName()  # noqa: F841 expected to fail
    print("✓ Tool correctly rejected")


@pytest.mark.asyncio
async def test_tool_missing_description():
    """Test that Tool subclass without description attribute fails."""
    print("\n=== Testing Tool Missing Description ===")

    # Define a tool without description attribute
    class ToolWithoutDescription(fm.Tool):
        name = "tool_without_description"

        @property
        def arguments_schema(self) -> fm.GenerationSchema:
            return CalculatorParams.generation_schema()

        async def call(self, args: fm.GeneratedContent) -> str:
            return "result"

    # Try to instantiate - should fail
    with pytest.raises(AssertionError):
        tool = ToolWithoutDescription()  # noqa: F841 expected to fail
    print("✓ Tool correctly rejected")


@pytest.mark.asyncio
async def test_tool_missing_arguments_schema():
    """Test that Tool subclass without arguments_schema property fails."""
    print("\n=== Testing Tool Missing Arguments Schema ===")

    # Define a tool without arguments_schema property
    class ToolWithoutSchema(fm.Tool):
        name = "tool_without_schema"
        description = "A tool without arguments schema"

        async def call(self, args: fm.GeneratedContent) -> str:
            return "result"

    # Try to instantiate - should fail
    with pytest.raises((AssertionError, TypeError)):
        tool = ToolWithoutSchema()  # type: ignore # noqa: F841 expected to fail
    print("✓ Tool correctly rejected")


@pytest.mark.asyncio
async def test_tool_missing_call_method():
    """Test that Tool subclass without call method fails."""
    print("\n=== Testing Tool Missing Call Method ===")

    # Define a tool without call method
    class ToolWithoutCall(fm.Tool):
        name = "tool_without_call"
        description = "A tool without call method"

        @property
        def arguments_schema(self) -> fm.GenerationSchema:
            return CalculatorParams.generation_schema()

    # Try to instantiate - should fail
    with pytest.raises((AssertionError, TypeError)):
        tool = ToolWithoutCall()  # type: ignore # noqa: F841 expected to fail
    print("✓ Tool correctly rejected")


@pytest.mark.asyncio
async def test_tool_invalid_name_type():
    """Test that Tool subclass with non-string name fails."""
    print("\n=== Testing Tool Invalid Name Type ===")

    # Define a tool with non-string name
    class ToolWithInvalidName(fm.Tool):
        name = 123  # type: ignore # Should be a string
        description = "A tool with invalid name type"

        @property
        def arguments_schema(self) -> fm.GenerationSchema:
            return CalculatorParams.generation_schema()

        async def call(self, args: fm.GeneratedContent) -> str:
            return "result"

    # Try to instantiate - should fail
    with pytest.raises(TypeError):
        tool = ToolWithInvalidName()  # noqa: F841 expected to fail
    print("✓ Tool correctly rejected")


@pytest.mark.asyncio
async def test_tool_invalid_description_type():
    """Test that Tool subclass with non-string description fails."""
    print("\n=== Testing Tool Invalid Description Type ===")

    # Define a tool with non-string description
    class ToolWithInvalidDescription(fm.Tool):
        name = "tool_with_invalid_description"
        description = ["not", "a", "string"]  # type: ignore # Should be a string

        @property
        def arguments_schema(self) -> fm.GenerationSchema:
            return CalculatorParams.generation_schema()

        async def call(self, args: fm.GeneratedContent) -> str:
            return "result"

    # Try to instantiate - should fail
    with pytest.raises(TypeError):
        tool = ToolWithInvalidDescription()  # noqa: F841 expected to fail
    print("✓ Tool correctly rejected")


@pytest.mark.asyncio
async def test_tool_invalid_schema_type():
    """Test that Tool subclass with invalid arguments_schema type fails."""
    print("\n=== Testing Tool Invalid Schema Type ===")

    # Define a tool with invalid schema type
    class ToolWithInvalidSchema(fm.Tool):
        name = "tool_with_invalid_schema"
        description = "A tool with invalid schema type"

        @property
        def arguments_schema(self):  # type: ignore we know it's wrong, that's the point
            return {"not": "a", "schema": "object"}  # Should be GenerationSchema

        async def call(self, args: fm.GeneratedContent) -> str:
            return "result"

    # Try to instantiate - should fail
    with pytest.raises(TypeError):
        tool = ToolWithInvalidSchema()  # noqa: F841 expected to fail
    print("✓ Tool correctly rejected")


@pytest.mark.asyncio
async def test_tool_non_async_call():
    """Test that Tool subclass with non-async call method fails."""
    print("\n=== Testing Tool Non-Async Call ===")

    # Define a tool with non-async call method
    class ToolWithNonAsyncCall(fm.Tool):
        name = "tool_with_non_async_call"
        description = "A tool with non-async call method"

        @property
        def arguments_schema(self) -> fm.GenerationSchema:
            return CalculatorParams.generation_schema()

        def call(self, args: fm.GeneratedContent) -> str:  # type: ignore # Not async!
            return "result"

    # Try to instantiate - should fail
    with pytest.raises(TypeError):
        tool = ToolWithNonAsyncCall()  # noqa: F841 expected to fail
    print("✓ Tool correctly rejected")


@pytest.mark.asyncio
async def test_parallel_tool_calling():
    """Test parallel execution of tool calls."""
    print("\n=== Testing Parallel Tool Calling ===")

    calc_tool = SimpleCalculatorTool()

    # Create multiple tool call tasks
    tasks = [
        calc_tool.call(
            fm.GeneratedContent(
                content_dict={"operation": "add", "a": float(i), "b": 1.0}
            )
        )
        for i in range(5)
    ]

    import time

    start_time = time.time()
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start_time

    assert len(results) == 5, f"Expected 5 results, got {len(results)}"
    for i, result in enumerate(results):
        expected = str(float(i + 1))
        assert expected in result, f"Expected {expected} in result {i}: {result}"

    print(f"✓ Parallel tool calls completed in {elapsed:.2f}s")


@pytest.mark.asyncio
async def test_session_with_parallel_tool_calls():
    """Test session with parallel tool calls."""
    print("\n=== Testing Session with Parallel Tool Calls ===")
    # Track tool call counts
    call_count = {"calc": 0, "user": 0}

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if not is_available:
        pytest.skip(f"No model available: {reason}")

    call_count = {"calc": 0, "user": 0}

    class TrackingCalculatorTool(fm.Tool):
        name = "calculator"
        description = "Perform arithmetic operations"

        @property
        def arguments_schema(self) -> fm.GenerationSchema:
            return CalculatorParams.generation_schema()

        async def call(self, args: fm.GeneratedContent) -> str:
            call_count["calc"] += 1
            operation = args.value(str, for_property="operation")
            a = args.value(float, for_property="a")
            b = args.value(float, for_property="b")
            if operation == "add":
                result = a + b
            elif operation == "multiply":
                result = a * b
            else:
                result = 0
            return f"{result}"

    # Create tracking user info tool
    class TrackingUserTool(fm.Tool):
        name = "get_user"
        description = "Get user information"

        @property
        def arguments_schema(self) -> fm.GenerationSchema:
            return UserInfoParams.generation_schema()

        async def call(self, args: fm.GeneratedContent) -> str:
            call_count["user"] += 1
            user_id = args.value(int, for_property="user_id")
            return f'{{"id": {user_id}, "name": "User{user_id}"}}'

    calc_tool = TrackingCalculatorTool()
    user_tool = TrackingUserTool()

    # Create session with tracking tools
    session = fm.LanguageModelSession(
        instructions="You are a helpful assistant. Use the provided tools to answer questions.",
        model=model,
        tools=[calc_tool, user_tool],
    )

    print(f"✓ Created session with {len([calc_tool, user_tool])} tracking tools")

    # Request that requires multiple tool calls with timeout to prevent hanging
    try:
        response = await asyncio.wait_for(
            session.respond(
                "What is 5 + 3 and 4 * 2? Also get info for user 1 and user 2."
            ),
            timeout=30.0,  # 30 second timeout to prevent indefinite hanging
        )

        # Verify response was generated
        assert response, "Expected non-empty response"
        print(f"✓ Got response: {response[:100]}{'...' if len(response) > 100 else ''}")

        # Report tool call counts
        total_calls = call_count["calc"] + call_count["user"]
        print(
            f"✓ Tools called {total_calls} times: calculator={call_count['calc']}, user={call_count['user']}"
        )
    except asyncio.TimeoutError:
        # If timeout occurs, wait for session to finish responding
        print("⚠ Request timed out after 30 seconds")
        while session.is_responding:
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.2)  # Additional delay for native cleanup
        pytest.fail(
            "Session response timed out - possible infinite tool calling loop or model issue"
        )
