# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Test Foundation Models guided generation from JSON schemas
"""

import apple_fm_sdk as fm
import pytest
import json
from tester_schemas.validate_schemas import (
    validate_age,
    validate_cat,
    validate_hedgehog,
    validate_person,
    validate_shelter,
    validate_pet_club,
    validate_newsletter,
)


@pytest.mark.asyncio
async def test_age_schema(model):
    """
    Test using age.json schema.

    Schema aspects covered:
    - Basic object with simple integer properties
    - Required properties (years, months)
    - additionalProperties: false (strict schema)
    - x-order property ordering
    """
    print("\n=== Testing age.json Schema ===")
    print(
        "Schema aspects: Basic object, required integer properties, strict schema, property ordering"
    )
    fileName = "tests/tester_schemas/age.json"
    with open(fileName, "r") as file:
        schema = json.load(file)

    # Get the default model
    session = fm.LanguageModelSession(model=model)

    # Use the schema directly
    generated_content = await session.respond(
        "Generate an elderly cat who likes yarn",
        json_schema=schema,
    )
    print(f"✓ Got generated content: {generated_content}")

    # Validate using the validator
    validate_age(generated_content)
    print("✓ Age validation passed")


@pytest.mark.asyncio
async def test_cat_schema(model):
    """
    Test using cat.json schema.

    Schema aspects covered:
    - Object with mixed property types (string, nested object reference)
    - $defs for reusable schema definitions
    - $ref to reference definitions within the schema
    - Property descriptions for semantic guidance
    - Nested object structure (Age within Cat)
    """
    print("\n=== Testing cat.json Schema ===")
    print(
        "Schema aspects: Mixed property types, $defs/$ref for nested objects, property descriptions"
    )
    fileName = "tests/tester_schemas/cat.json"
    with open(fileName, "r") as file:
        schema = json.load(file)

    # Get the default model
    session = fm.LanguageModelSession(model=model)

    # Use the schema directly
    generated_content = await session.respond(
        "Generate an elderly cat who likes yarn",
        json_schema=schema,
    )
    print(f"✓ Got generated content: {generated_content}")

    # Validate using the validator
    validate_cat(generated_content)
    print("✓ Cat validation passed")


@pytest.mark.asyncio
async def test_hedgehog_schema(model):
    """
    Test using hedgehog.json schema.

    Schema aspects covered:
    - Numeric constraints: minimum/maximum for integer properties
    - Enum constraints for string properties (favoriteFood, home)
    - Array properties with items of a specific type
    - Array size constraints: minItems/maxItems (exactly 3 hobbies)
    - Multiple constraint types in a single schema
    """
    print("\n=== Testing hedgehog.json Schema ===")
    print("Schema aspects: Numeric min/max, enum constraints, arrays with size limits")
    fileName = "tests/tester_schemas/hedgehog.json"
    with open(fileName, "r") as file:
        schema = json.load(file)

    # Get the default model
    session = fm.LanguageModelSession(model=model)

    # Use the schema directly
    generated_content = await session.respond(
        "Generate a very old hedgehog who likes to dance",
        json_schema=schema,
    )
    print(f"✓ Got generated content: {generated_content}")

    # Validate using the validator
    validate_hedgehog(generated_content)
    print("✓ Hedgehog validation passed")


@pytest.mark.asyncio
async def test_person_schema(model):
    """
    Test using person.json schema.

    Schema aspects covered:
    - Recursive schema: $ref to self (Person contains array of Person)
    - Mix of required and optional properties (age is optional)
    - Array with complex object items (children array)
    - Array size constraint with maxItems
    - Self-referential data structures
    """
    print("\n=== Testing person.json Schema ===")
    print(
        "Schema aspects: Recursive self-reference, optional properties, arrays of complex objects"
    )
    fileName = "tests/tester_schemas/person.json"
    with open(fileName, "r") as file:
        schema = json.load(file)

    # Get the default model
    session = fm.LanguageModelSession(model=model)

    try:
        # Use the schema directly
        generated_content = await session.respond(
            "Generate an elderly inn keeper character who has 5 children and NO grandchildren",
            json_schema=schema,
        )
        print(f"✓ Got generated content: {generated_content}")

        # Validate using the validator
        validate_person(generated_content)
        print("✓ Person validation passed")

        # Prompt-specific validation: the prompt asks for 5 children, but schema maxItems is 3
        children = generated_content.value(list, for_property="children")
        assert len(children) == 3, (
            f"✗ Generated wrong number of children: {len(children)} instead of expected 3 (schema constraint)"
        )
        print("✓ Correctly limited to 3 children despite prompt asking for 5")

    # This sometimes happens but its not an issue with the schema handling itself
    except fm.ExceededContextWindowSizeError as e:
        print(f"✗ Test skipped due to context window size error: {e}")
        raise


@pytest.mark.asyncio
async def test_shelter_schema(model):
    """
    Test using shelter.json schema.

    Schema aspects covered:
    - Arrays of complex objects with nested references
    - Multiple levels of $defs (Age, Cat definitions)
    - Combining arrays with $ref to complex types
    - Schema composition with nested object definitions
    """
    print("\n=== Testing shelter.json Schema ===")
    print(
        "Schema aspects: Arrays of complex objects, multi-level $defs, nested references"
    )
    fileName = "tests/tester_schemas/shelter.json"
    with open(fileName, "r") as file:
        schema = json.load(file)

    # Get the default model
    session = fm.LanguageModelSession(model=model)

    # Use the schema directly
    generated_content = await session.respond(
        "Generate a shelter with 3 cats: a shorthair named Whiskers, a longhair named Fluffy, and a hairless named Sphinx",
        json_schema=schema,
    )
    print(f"✓ Got generated content: {generated_content}")

    # Validate using the validator
    validate_shelter(generated_content)
    print("✓ Shelter validation passed")


@pytest.mark.asyncio
async def test_pet_club_schema(model):
    """
    Test using petClub.json schema.

    Schema aspects covered:
    - Complex schema with multiple entity types (Person, Cat, Hedgehog)
    - Multiple arrays of different complex types in one schema
    - Mixing simple arrays (otherPets: string[]) with complex object arrays
    - Reusing definitions across multiple properties
    - Comprehensive integration of all constraint types
    """
    print("\n=== Testing petClub.json Schema ===")
    print(
        "Schema aspects: Multiple entity types, mixed array types, comprehensive constraint integration"
    )
    fileName = "tests/tester_schemas/petClub.json"
    with open(fileName, "r") as file:
        schema = json.load(file)

    # Get the default model
    session = fm.LanguageModelSession(model=model)

    # Use the schema directly with a comprehensive prompt
    generated_content = await session.respond(
        "Generate a pet club with 2 members (Alice age 25 and Bob age 30), "
        "2 cats (a shorthair named Mittens and a longhair named Fluffy), "
        "1 hedgehog (named Spike who is 5 years old and loves carrots), "
        "2 other pets (a parrot and a turtle), "
        "and Alice as the president",
        json_schema=schema,
    )
    print(f"✓ Got generated content: {generated_content}")

    # Validate using the validator
    validate_pet_club(generated_content)
    print("✓ Pet club validation passed")


@pytest.mark.asyncio
async def test_newsletter_schema(model):
    """
    Test using newsletter.json schema.

    Schema aspects covered:
    - Mix of required and optional properties (only title/topic required)
    - Multiple optional complex object properties
    - Optional arrays of complex objects (featuredCats, featuredStaff)
    - Optional single complex object (featuredHedgehog)
    - Flexible schema allowing partial data generation
    - Real-world use case: content with variable sections
    """
    print("\n=== Testing newsletter.json Schema ===")
    print(
        "Schema aspects: Required vs optional properties, optional complex objects/arrays, flexible schema"
    )
    fileName = "tests/tester_schemas/newsletter.json"
    with open(fileName, "r") as file:
        schema = json.load(file)

    # Get the default model
    session = fm.LanguageModelSession(model=model)

    # Use the schema directly with a comprehensive prompt
    generated_content = await session.respond(
        "Generate a newsletter featuring senior cats available for adoption! \
        cats available for adoption! \
        - Do not mention any Hedgehogs. \
        - Mention 3 staff members who love senior cats \
        - This article does NOT have a sponsor",
        json_schema=schema,
    )
    print(f"✓ Got generated content: {generated_content}")

    # Validate using the validator
    validate_newsletter(generated_content)
    print("✓ Newsletter validation passed")

    # Prompt-specific validations
    sponsor = generated_content.value(str, for_property="sponsor")
    assert sponsor is None, "✗ Sponsor should be None as per prompt"
    print("✓ Correctly omitted sponsor as requested")

    featuredHedgehog = generated_content.value(dict, for_property="featuredHedgehog")
    assert featuredHedgehog is None, "✗ Featured hedgehog should be None as per prompt"
    print("✓ Correctly omitted hedgehog as requested")

    featuredStaff = generated_content.value(list, for_property="featuredStaff")
    if featuredStaff is not None:
        assert len(featuredStaff) == 3, (
            f"✗ Featured staff count incorrect: {len(featuredStaff)} instead of 3"
        )
        print("✓ Correctly included 3 staff members as requested")
