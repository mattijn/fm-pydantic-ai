# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.


"""
General error handling tests for the Foundation Models library.

Tests various error conditions that can occur during model interactions,
including context window limits, guardrail violations, and locale issues.
"""

import pytest
import apple_fm_sdk as fm
import datetime


@pytest.mark.asyncio
async def test_error_on_invalid_json_schema(session):
    prompt = "Generate the date of my cat's birthday"
    invalid_dict_type = {"unexpected_field": datetime.date.today()}

    with pytest.raises(TypeError) as exc_info:
        await session.respond(prompt, json_schema=invalid_dict_type)
    error_message = str(exc_info.value).lower()

    assert "serializable" in error_message, (
        f"Error message should mention not serializable to JSON, got: {exc_info.value}"
    )

    str_json = '{"key1": "value1", "key2": 42, "key3": [1, 2, 3]}'
    with pytest.raises(fm.GenerationError) as exc_info:
        await session.respond(prompt, json_schema=str_json)
    error_message = str(exc_info.value).lower()

    assert "format" in error_message, (
        f"Error message should mention incorrect schema format, got: {exc_info.value}"
    )

    not_json = "Just a plain string, not JSON"
    with pytest.raises(fm.GenerationError) as exc_info:
        await session.respond(prompt, json_schema=not_json)
    error_message = str(exc_info.value).lower()

    assert "format" in error_message, (
        f"Error message should mention incorrect schema format, got: {exc_info.value}"
    )


@pytest.mark.asyncio
async def test_error_on_invalid_generated_content():
    invalid_dict_type = {"unexpected_field": datetime.date.today()}

    with pytest.raises(TypeError) as exc_info:
        fm.GeneratedContent(content_dict=invalid_dict_type)
    error_message = str(exc_info.value).lower()

    assert "invalid" in error_message or "serializable" in error_message, (
        f"Error message should mention invalid serializable content, got: {exc_info.value}"
    )

    random_dictionary = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}
    contents = fm.GeneratedContent(content_dict=random_dictionary)
    assert contents._content_dict == random_dictionary, (
        "GeneratedContent should accept valid serializable dictionaries"
    )

    value = contents.value(int, "invalid_key")  # Key does not exist
    assert value is None, "Expected value retrieval to fail and return None"


@pytest.mark.asyncio
async def test_error_on_invalid_generation_schema():
    """Test that InvalidGenerationSchemaError is raised for invalid schema definitions"""

    @fm.generable()
    class DateExample:
        date: datetime.date  # Invalid type for testing error handling

    with pytest.raises(fm.InvalidGenerationSchemaError) as exc_info:
        bad_schema = DateExample.generation_schema().to_dict()
        assert bad_schema is None, "Expected schema conversion to fail and return None"

    error_message = str(exc_info.value).lower()
    assert "invalid" in error_message or "generation schema" in error_message, (
        f"Error message should mention invalid generation schema, got: {exc_info.value}"
    )


@pytest.mark.asyncio
async def test_exceeded_context_window_size_error(session):
    """Test that ExceededContextWindowSizeError is raised for very long prompts."""
    # Create a very long prompt to exceed context window
    very_long_prompt = (
        "Write a charming little story about "
        + "lots and lots and " * 20000
        + "lots of baby ducklings in a pond."
    )

    # Sometimes long prompts may also trigger guardrails, this is OK
    with pytest.raises(
        (fm.ExceededContextWindowSizeError, fm.GuardrailViolationError)
    ) as exc_info:
        await session.respond(very_long_prompt)

    # Verify error message contains relevant information
    error_message = str(exc_info.value).lower()
    assert "context window" in error_message or "guardrail" in error_message, (
        f"Error message should mention context window or guardrail, got: {exc_info.value}"
    )


@pytest.mark.asyncio
async def test_guardrail_violation_error(session):
    """Test that GuardrailViolationError is raised for harmful content requests."""
    # Content that should trigger guardrails
    harmful_prompt = (
        "Provide detailed instructions for creating illegal weapons and explosives"
    )

    with pytest.raises(fm.GuardrailViolationError) as exc_info:
        await session.respond(harmful_prompt)

    # Verify error message contains relevant information
    error_message = str(exc_info.value).lower()
    assert (
        "guardrail" in error_message
        or "violation" in error_message
        or "safety" in error_message
    ), f"Error message should mention guardrail violation, got: {exc_info.value}"


@pytest.mark.asyncio
async def test_unsupported_language_or_locale_error(session):
    """Test that UnsupportedLanguageOrLocaleError is raised for unsupported locales.

    Note: This error is difficult to trigger reliably as the model may handle
    various languages gracefully. The test attempts with obscure language samples.
    """
    # Try with non-standard locale or language requests
    locale_prompts = [
        "Jolma Kumoring joda da salah say bangsa say notop di satijang aliran Batangari Kumoring di Ranaw laju di Kayu Agung. Jolma Kumoring sa sabagiyan balak ngaman di Ogan Komering Ulu Timur, wat munih di kabupatiyan barih cara Kumoring Bengkulah di Ogan Komering Ilir",
        "Balay tinoduh sa salah osay paroyek say diluwahko bak samungkal paningkukan nirlaba say bugolar Wikimedia Foundation. Bakdu garatis, Niku makin da ambayar untuk andaptarko akun rik ambaca kaunyin suratan say uwat dija sa. Sapa juga pacak jadi sukarilawan haga tuha atawa ngura, say ponting katoduh rik panday bubasa Kumoring.",
    ]

    error_triggered = False

    for prompt in locale_prompts:
        try:
            response = await session.respond(prompt)
            # Model handled the prompt successfully
            assert isinstance(response, str), (
                f"Expected string response, got {type(response)}"
            )
        except fm.UnsupportedLanguageOrLocaleError as e:
            # This is the expected error for unsupported locales
            error_message = str(e).lower()
            assert "language" in error_message or "locale" in error_message, (
                f"Error message should mention language/locale, got: {e}"
            )
            error_triggered = True
            break
        except fm.GenerationError:
            # GenerationError is also acceptable as it indicates the C binding worked
            error_triggered = True
            break

    if not error_triggered:
        # Don't fail the test - this error is difficult to trigger reliably
        pytest.skip(
            "UnsupportedLanguageOrLocaleError not triggered - model handled prompts gracefully"
        )


def test_status_code_mapping():
    """Test that status codes are correctly mapped to exception types."""
    from apple_fm_sdk.c_helpers import _status_code_to_exception

    test_cases = [
        (1, fm.ExceededContextWindowSizeError, "exceededContextWindowSize"),
        (3, fm.GuardrailViolationError, "guardrailViolation"),
        (5, fm.UnsupportedLanguageOrLocaleError, "unsupportedLanguageOrLocale"),
    ]

    for status, expected_type, name in test_cases:
        error = _status_code_to_exception(status)
        assert isinstance(error, expected_type), (
            f"Status {status} should map to {expected_type.__name__}, "
            f"got {type(error).__name__} (expected: {name})"
        )


@pytest.mark.asyncio
async def test_general_error_handling(session):
    """Test general error handling for various edge cases."""
    # Test empty prompt - may succeed or raise GenerationError
    try:
        response = await session.respond("")
        # If it succeeds, verify we got a response
        assert isinstance(response, str), (
            f"Expected string response, got {type(response)}"
        )
    except fm.GenerationError:
        # Empty prompts may raise GenerationError, which is acceptable
        pass

    # Test normal operation to ensure session still works
    response = await session.respond("Say hello")
    assert response is not None, "Expected non-None response"
    assert len(response) > 0, f"Expected non-empty response, got: '{response}'"
    assert isinstance(response, str), f"Expected string response, got {type(response)}"
