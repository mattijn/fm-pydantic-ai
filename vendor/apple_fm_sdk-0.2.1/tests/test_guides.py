# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Test Foundation Models GenerationGuide - comprehensive testing of all guide types
"""

import apple_fm_sdk as fm
from typing import List
import pytest
import re


# Test class for anyOf guide
@fm.generable("Product with anyOf constraint")
class ProductAnyOfGuide:
    category: str = fm.guide(
        "Product category",
        anyOf=["electronics", "clothing", "books", "home", "sports"],
    )
    status: str = fm.guide(
        "Product status", anyOf=["available", "out_of_stock", "discontinued"]
    )


# Test class for constant guide
@fm.generable("Product with constant constraints")
class ProductConstantGuide:
    status: str = fm.guide("Product status", constant="active")
    version: str = fm.guide("API version", constant="v1.0")
    category: str = fm.guide("Product category", constant="electronics")


# Test class for count guide
@fm.generable("Product with count constraints")
class ProductCountGuide:
    features: List[str] = fm.guide("Product features", count=3)
    tags: List[str] = fm.guide("Product tags", count=5)
    colors: List[str] = fm.guide("Available colors", count=2)


# Test class for element guide
@fm.generable("Product with element constraints")
class ProductElementGuide:
    ratings: List[int] = fm.guide(
        "Product ratings", element=fm.GenerationGuide.range((1, 5))
    )
    prices: List[float] = fm.guide(
        "Historical prices", element=fm.GenerationGuide.minimum(0.01)
    )
    categories: List[str] = fm.guide(
        "Product categories",
        element=fm.GenerationGuide.anyOf(["tech", "home", "sports"]),
    )


# Test class for max_items guide
@fm.generable("Product with max_items constraints")
class ProductMaxItemsGuide:
    features: List[str] = fm.guide("Product features", max_items=5)
    tags: List[str] = fm.guide("Product tags", max_items=3)
    colors: List[str] = fm.guide("Product colors", max_items=4)


# Test class for maximum guide
@fm.generable("Product with maximum constraints")
class ProductMaximumGuide:
    price: float = fm.guide(maximum=500.0)
    weight: float = fm.guide("Product weight in kg", maximum=10.0)
    quantity: int = fm.guide("Available quantity", maximum=100)


# Test class for min_items guide
@fm.generable("Product with min_items constraints")
class ProductMinItemsGuide:
    features: List[str] = fm.guide("Product features", min_items=2)
    tags: List[str] = fm.guide("Product tags", min_items=1)
    reviews: List[str] = fm.guide("Customer reviews", min_items=3)


# Test class for minimum guide
@fm.generable("Product with minimum constraints")
class ProductMinimumGuide:
    price: float = fm.guide(minimum=1.0)
    weight: float = fm.guide("Product weight in kg", minimum=0.1)
    quantity: int = fm.guide("Available quantity", minimum=1)


# Test class for range guide
@fm.generable("Product with range constraints")
class ProductRangeGuide:
    price: float = fm.guide(range=(0.99, 999.99))
    rating: float = fm.guide("Product rating", range=(1.0, 5.0))
    discount_percent: int = fm.guide("Discount percentage", range=(0, 100))


# Test class for regex guide
@fm.generable("Product with regex constraints")
class ProductRegexGuide:
    name: str = fm.guide("Name", regex=r"\w")
    product_code: str = fm.guide("Product code", regex=r"\d+")


# Test class combining multiple guide types
@fm.generable("Product with combined constraints")
class ProductCombinedGuides:
    name: str = fm.guide("Product name")
    category: str = fm.guide(
        "Product category", anyOf=["electronics", "clothing", "books"]
    )
    price: float = fm.guide("Product price", range=(10.0, 1000.0))
    features: List[str] = fm.guide("Product features", count=3)
    rating: float = fm.guide("Product rating", minimum=1.0, maximum=5.0)
    tags: List[str] = fm.guide("Product tags", min_items=1, max_items=5)


@pytest.mark.asyncio
async def test_anyOf_guide(model):
    """Test anyOf guide constraint."""
    print("\n=== Testing anyOf Guide ===")

    session = fm.LanguageModelSession(
        "You are a helpful product assistant.", model=model
    )

    result = await session.respond(
        "Generate a laptop product with appropriate category and status.",
        generating=ProductAnyOfGuide,
    )

    assert type(result) is ProductAnyOfGuide, (
        f"✗ Invalid generated content type: {type(result)}"
    )

    # Validate anyOf constraints
    valid_categories = ["electronics", "clothing", "books", "home", "sports"]
    valid_statuses = ["available", "out_of_stock", "discontinued"]

    assert result.category in valid_categories, (
        f"✗ Invalid category: {result.category}. Must be one of {valid_categories}"
    )
    assert result.status in valid_statuses, (
        f"✗ Invalid status: {result.status}. Must be one of {valid_statuses}"
    )

    print(f"✓ Category constraint satisfied: {result.category}")
    print(f"✓ Status constraint satisfied: {result.status}")


@pytest.mark.asyncio
async def test_range_guide(model):
    """Test range guide constraint."""
    print("\n=== Testing Range Guide ===")

    session = fm.LanguageModelSession(
        "You are a helpful product assistant.", model=model
    )

    result = await session.respond(
        "Generate a super expensive product with a super high rating and an enourmos discount.",
        generating=ProductRangeGuide,
    )

    assert type(result) is ProductRangeGuide, (
        f"✗ Invalid generated content type: {type(result)}"
    )

    # Validate range constraints
    assert 0.99 <= result.price <= 999.99, (
        f"✗ Price out of range: {result.price}. Must be between 0.99 and 999.99"
    )
    assert 1.0 <= result.rating <= 5.0, (
        f"✗ Rating out of range: {result.rating}. Must be between 1.0 and 5.0"
    )
    assert 0 <= result.discount_percent <= 100, (
        f"✗ Discount out of range: {result.discount_percent}. Must be between 0 and 100"
    )

    print(f"✓ Price range constraint satisfied: ${result.price}")
    print(f"✓ Rating range constraint satisfied: {result.rating}")
    print(f"✓ Discount range constraint satisfied: {result.discount_percent}%")


@pytest.mark.asyncio
async def test_count_guide(model):
    """Test count guide constraint."""
    print("\n=== Testing Count Guide ===")

    session = fm.LanguageModelSession(
        "You are a helpful product assistant.", model=model
    )

    result = await session.respond(
        "Generate a smartphone.",
        generating=ProductCountGuide,
    )

    assert type(result) is ProductCountGuide, (
        f"✗ Invalid generated content type: {type(result)}"
    )

    # Validate count constraints
    assert len(result.features) == 3, (
        f"✗ Wrong feature count: {len(result.features)}. Must be exactly 3"
    )
    assert len(result.tags) == 5, (
        f"✗ Wrong tag count: {len(result.tags)}. Must be exactly 5"
    )
    assert len(result.colors) == 2, (
        f"✗ Wrong color count: {len(result.colors)}. Must be exactly 2"
    )

    print(f"✓ Features count constraint satisfied: {len(result.features)} items")
    print(f"✓ Tags count constraint satisfied: {len(result.tags)} items")
    print(f"✓ Colors count constraint satisfied: {len(result.colors)} items")


@pytest.mark.asyncio
async def test_minimum_guide(model):
    """Test minimum guide constraint."""
    print("\n=== Testing Minimum Guide ===")

    session = fm.LanguageModelSession(
        "You are a helpful product assistant.", model=model
    )

    result = await session.respond(
        "Generate an ultra cheap lightweight product",
        generating=ProductMinimumGuide,
    )

    assert type(result) is ProductMinimumGuide, (
        f"✗ Invalid generated content type: {type(result)}"
    )

    # Validate minimum constraints
    assert result.price >= 1.0, f"✗ Price below minimum: {result.price}. Must be >= 1.0"
    assert result.weight >= 0.1, (
        f"✗ Weight below minimum: {result.weight}. Must be >= 0.1"
    )
    assert result.quantity >= 1, (
        f"✗ Quantity below minimum: {result.quantity}. Must be >= 1"
    )

    print(f"✓ Price minimum constraint satisfied: ${result.price}")
    print(f"✓ Weight minimum constraint satisfied: {result.weight}kg")
    print(f"✓ Quantity minimum constraint satisfied: {result.quantity}")


@pytest.mark.asyncio
async def test_maximum_guide(model):
    """Test maximum guide constraint."""
    print("\n=== Testing Maximum Guide ===")

    session = fm.LanguageModelSession(
        "You are a helpful product assistant.", model=model
    )

    result = await session.respond(
        "Generate a ton of super expensive and heavy products.",
        generating=ProductMaximumGuide,
    )

    assert type(result) is ProductMaximumGuide, (
        f"✗ Invalid generated content type: {type(result)}"
    )

    # Validate maximum constraints
    assert result.price <= 500.0, (
        f"✗ Price above maximum: {result.price}. Must be <= 500.0"
    )
    assert result.weight <= 10.0, (
        f"✗ Weight above maximum: {result.weight}. Must be <= 10.0"
    )
    assert result.quantity <= 100, (
        f"✗ Quantity above maximum: {result.quantity}. Must be <= 100"
    )

    print(f"✓ Price maximum constraint satisfied: ${result.price}")
    print(f"✓ Weight maximum constraint satisfied: {result.weight}kg")
    print(f"✓ Quantity maximum constraint satisfied: {result.quantity}")


@pytest.mark.asyncio
async def test_min_items_guide(model):
    """Test min_items guide constraint."""
    print("\n=== Testing Min Items Guide ===")

    session = fm.LanguageModelSession(
        "You are a helpful product assistant.", model=model
    )

    result = await session.respond(
        "Generate a cool product.",
        generating=ProductMinItemsGuide,
    )

    assert type(result) is ProductMinItemsGuide, (
        f"✗ Invalid generated content type: {type(result)}"
    )

    # Validate min_items constraints
    assert len(result.features) >= 2, (
        f"✗ Too few features: {len(result.features)}. Must have at least 2"
    )
    assert len(result.tags) >= 1, (
        f"✗ Too few tags: {len(result.tags)}. Must have at least 1"
    )
    assert len(result.reviews) >= 3, (
        f"✗ Too few reviews: {len(result.reviews)}. Must have at least 3"
    )

    print(f"✓ Features min_items constraint satisfied: {len(result.features)} items")
    print(f"✓ Tags min_items constraint satisfied: {len(result.tags)} items")
    print(f"✓ Reviews min_items constraint satisfied: {len(result.reviews)} items")


@pytest.mark.asyncio
async def test_max_items_guide(model):
    """Test max_items guide constraint."""
    print("\n=== Testing Max Items Guide ===")

    session = fm.LanguageModelSession(
        "You are a helpful product assistant.", model=model
    )

    # Note here the prompt requests more than the max_items limits
    result = await session.respond(
        "Generate a product with many features, at 5 tags, and many colors.",
        generating=ProductMaxItemsGuide,
    )

    assert type(result) is ProductMaxItemsGuide, (
        f"✗ Invalid generated content type: {type(result)}"
    )

    # Validate max_items constraints
    assert len(result.features) <= 5, (
        f"✗ Too many features: {len(result.features)}. Must have at most 5"
    )
    assert len(result.tags) <= 3, (
        f"✗ Too many tags: {len(result.tags)}. Must have at most 3"
    )
    assert len(result.colors) <= 4, (
        f"✗ Too many colors: {len(result.colors)}. Must have at most 4"
    )

    print(f"✓ Features max_items constraint satisfied: {len(result.features)} items")
    print(f"✓ Tags max_items constraint satisfied: {len(result.tags)} items")
    print(f"✓ Colors max_items constraint satisfied: {len(result.colors)} items")


@pytest.mark.asyncio
async def test_regex_guide(model):
    """Test regex guide constraint."""
    print("\n=== Testing Regex Guide ===")

    session = fm.LanguageModelSession(
        "You are a helpful assistant that generates valid contact information.",
        model=model,
    )

    result = await session.respond(
        "Generate contact information with valid email, phone number, and product code formats.",
        generating=ProductRegexGuide,
    )

    assert type(result) is ProductRegexGuide, (
        f"✗ Invalid generated content type: {type(result)}"
    )

    # Validate regex constraints
    name_pattern = r"\w"
    code_pattern = r"\d+"

    assert re.match(name_pattern, result.name), (
        f"✗ Invalid name format: {result.name}. Must match pattern {name_pattern}"
    )
    assert re.match(code_pattern, result.product_code), (
        f"✗ Invalid product code format: {result.product_code}. Must match pattern {code_pattern}"
    )

    print(f"✓ Name regex constraint satisfied: {result.name}")
    print(f"✓ Product code regex constraint satisfied: {result.product_code}")


@pytest.mark.asyncio
async def test_constant_guide(model):
    """Test constant guide constraint."""
    print("\n=== Testing Constant Guide ===")

    session = fm.LanguageModelSession(
        "You are a helpful product assistant.", model=model
    )

    result = await session.respond(
        "Generate a product with the specified constant values.",
        generating=ProductConstantGuide,
    )

    assert type(result) is ProductConstantGuide, (
        f"✗ Invalid generated content type: {type(result)}"
    )

    # Validate constant constraints
    assert result.status == "active", (
        f"✗ Invalid status: {result.status}. Must be exactly 'active'"
    )
    assert result.version == "v1.0", (
        f"✗ Invalid version: {result.version}. Must be exactly 'v1.0'"
    )
    assert result.category == "electronics", (
        f"✗ Invalid category: {result.category}. Must be exactly 'electronics'"
    )

    print(f"✓ Status constant constraint satisfied: {result.status}")
    print(f"✓ Version constant constraint satisfied: {result.version}")
    print(f"✓ Category constant constraint satisfied: {result.category}")


@pytest.mark.asyncio
async def test_element_guide(model):
    """Test element guide constraint."""
    print("\n=== Testing Element Guide ===")

    session = fm.LanguageModelSession(
        "You are a helpful product assistant.", model=model
    )

    result = await session.respond(
        "Generate a product with ratings (1-5), positive prices, and valid categories.",
        generating=ProductElementGuide,
    )

    assert type(result) is ProductElementGuide, (
        f"✗ Invalid generated content type: {type(result)}"
    )

    # Validate element constraints
    valid_categories = ["tech", "home", "sports"]

    # Check ratings are in range 1-5
    for rating in result.ratings:
        assert 1 <= rating <= 5, (
            f"✗ Rating out of range: {rating}. Must be between 1 and 5"
        )

    # Check prices are positive
    for price in result.prices:
        assert price >= 0.01, f"✗ Price below minimum: {price}. Must be >= 0.01"

    # Check categories are valid anyOf
    for category in result.categories:
        assert category in valid_categories, (
            f"✗ Invalid category: {category}. Must be one of {valid_categories}"
        )

    print(
        f"✓ Ratings element constraint satisfied: {len(result.ratings)} ratings in range 1-5"
    )
    print(f"✓ Prices element constraint satisfied: {len(result.prices)} prices >= 0.01")
    print(
        f"✓ Categories element constraint satisfied: {len(result.categories)} valid categories"
    )


@pytest.mark.asyncio
async def test_combined_guides(model):
    """Test multiple guide types combined in one class."""
    print("\n=== Testing Combined Guides ===")

    session = fm.LanguageModelSession(
        "You are a helpful product assistant.", model=model
    )

    result = await session.respond(
        "Generate a comprehensive product with all specified constraints.",
        generating=ProductCombinedGuides,
    )

    assert type(result) is ProductCombinedGuides, (
        f"✗ Invalid generated content type: {type(result)}"
    )

    # Validate all combined constraints
    valid_categories = ["electronics", "clothing", "books"]
    assert result.category in valid_categories, f"✗ Invalid category: {result.category}"

    assert 10.0 <= result.price <= 1000.0, f"✗ Price out of range: {result.price}"

    assert len(result.features) == 3, f"✗ Wrong feature count: {len(result.features)}"

    assert 1.0 <= result.rating <= 5.0, f"✗ Rating out of range: {result.rating}"

    assert 1 <= len(result.tags) <= 5, f"✗ Tags count out of range: {len(result.tags)}"

    print("✓ All combined constraints satisfied:")
    print(f"  - Category: {result.category}")
    print(f"  - Price: ${result.price}")
    print(f"  - Features count: {len(result.features)}")
    print(f"  - Rating: {result.rating}")
    print(f"  - Tags count: {len(result.tags)}")


@pytest.mark.asyncio
async def test_generation_guide_direct_creation():
    """Test creating GenerationGuide objects directly."""
    print("\n=== Testing Direct GenerationGuide Creation ===")

    # Test all guide creation methods
    choices_guide = fm.GenerationGuide.anyOf(["red", "green", "blue"])
    assert choices_guide.guide_type == fm.GuideType.anyOf
    assert choices_guide.value == ["red", "green", "blue"]
    print("✓ AnyOf guide created successfully")

    range_guide = fm.GenerationGuide.range((1, 10))
    assert range_guide.guide_type == fm.GuideType.range
    assert range_guide.value == (1, 10)
    print("✓ Range guide created successfully")

    count_guide = fm.GenerationGuide.count(5)
    assert count_guide.guide_type == fm.GuideType.count
    assert count_guide.value == 5
    print("✓ Count guide created successfully")

    minimum_guide = fm.GenerationGuide.minimum(0)
    assert minimum_guide.guide_type == fm.GuideType.minimum
    assert minimum_guide.value == 0
    print("✓ Minimum guide created successfully")

    maximum_guide = fm.GenerationGuide.maximum(100)
    assert maximum_guide.guide_type == fm.GuideType.maximum
    assert maximum_guide.value == 100
    print("✓ Maximum guide created successfully")

    min_items_guide = fm.GenerationGuide.min_items(2)
    assert min_items_guide.guide_type == fm.GuideType.minItems
    assert min_items_guide.value == 2
    print("✓ Min items guide created successfully")

    max_items_guide = fm.GenerationGuide.max_items(10)
    assert max_items_guide.guide_type == fm.GuideType.maxItems
    assert max_items_guide.value == 10
    print("✓ Max items guide created successfully")

    regex_guide = fm.GenerationGuide.regex(r"^[A-Z]{2}-[0-9]{4}$")
    assert regex_guide.guide_type == fm.GuideType.regex
    assert regex_guide.value == r"^[A-Z]{2}-[0-9]{4}$"
    print("✓ Regex guide created successfully")

    constant_guide = fm.GenerationGuide.constant("fixed_value")
    assert constant_guide.guide_type == fm.GuideType.constant
    assert constant_guide.value == "fixed_value"
    print("✓ Constant guide created successfully")

    element_guide = fm.GenerationGuide.element(fm.GenerationGuide.range((1, 10)))
    assert element_guide.guide_type == fm.GuideType.element
    assert isinstance(element_guide.value, fm.GenerationGuide)
    assert element_guide.value.guide_type == fm.GuideType.range
    print("✓ Element guide created successfully")


# Schema classes for UnsupportedGuide error scenarios


@fm.generable("Invalid: numeric guides on string")
class InvalidNumericGuidesOnString:
    # These should throw UnsupportedGuide errors
    name: str = fm.guide("Name with minimum", minimum=5)


@fm.generable("Invalid: numeric guides on string - maximum")
class InvalidMaximumOnString:
    name: str = fm.guide("Name with maximum", maximum=100)


@fm.generable("Invalid: numeric guides on string - range")
class InvalidRangeOnString:
    name: str = fm.guide("Name with range", range=(1, 10))


@fm.generable("Invalid: count guide on string")
class InvalidCountOnString:
    name: str = fm.guide("Name with count", count=5)


@fm.generable("Invalid: minItems guide on string")
class InvalidMinItemsOnString:
    name: str = fm.guide("Name with minItems", min_items=2)


@fm.generable("Invalid: maxItems guide on string")
class InvalidMaxItemsOnString:
    name: str = fm.guide("Name with maxItems", max_items=10)


@fm.generable("Invalid: anyOf guide on integer")
class InvalidAnyOfOnInteger:
    age: int = fm.guide("Age with anyOf", anyOf=["young", "old"])


@fm.generable("Invalid: regex guide on integer")
class InvalidRegexOnInteger:
    age: int = fm.guide("Age with regex", regex=r"\d+")


@fm.generable("Invalid: count guide on integer")
class InvalidCountOnInteger:
    age: int = fm.guide("Age with count", count=5)


@fm.generable("Invalid: minItems guide on integer")
class InvalidMinItemsOnInteger:
    age: int = fm.guide("Age with minItems", min_items=2)


@fm.generable("Invalid: maxItems guide on integer")
class InvalidMaxItemsOnInteger:
    age: int = fm.guide("Age with maxItems", max_items=10)


@fm.generable("Invalid: anyOf guide on float")
class InvalidAnyOfOnFloat:
    price: float = fm.guide("Price with anyOf", anyOf=["cheap", "expensive"])


@fm.generable("Invalid: regex guide on float")
class InvalidRegexOnFloat:
    price: float = fm.guide("Price with regex", regex=r"\d+\.\d+")


@fm.generable("Invalid: count guide on float")
class InvalidCountOnFloat:
    price: float = fm.guide("Price with count", count=3)


@fm.generable("Invalid: minItems guide on float")
class InvalidMinItemsOnFloat:
    price: float = fm.guide("Price with minItems", min_items=1)


@fm.generable("Invalid: maxItems guide on float")
class InvalidMaxItemsOnFloat:
    price: float = fm.guide("Price with maxItems", max_items=5)


@fm.generable("Invalid: anyOf guide on array of integers")
class InvalidAnyOfOnIntArray:
    scores: List[int] = fm.guide("Scores with anyOf", anyOf=["high", "low"])


@fm.generable("Invalid: regex guide on array of integers")
class InvalidRegexOnIntArray:
    scores: List[int] = fm.guide("Scores with regex", regex=r"\d+")


@fm.generable("Invalid: minimum guide on array of integers")
class InvalidMinimumOnIntArray:
    scores: List[int] = fm.guide("Scores with minimum", minimum=0)


@fm.generable("Invalid: maximum guide on array of integers")
class InvalidMaximumOnIntArray:
    scores: List[int] = fm.guide("Scores with maximum", maximum=100)


@fm.generable("Invalid: range guide on array of integers")
class InvalidRangeOnIntArray:
    scores: List[int] = fm.guide("Scores with range", range=(0, 100))


@fm.generable("Invalid: anyOf guide on array of floats")
class InvalidAnyOfOnFloatArray:
    prices: List[float] = fm.guide("Prices with anyOf", anyOf=["low", "high"])


@fm.generable("Invalid: regex guide on array of floats")
class InvalidRegexOnFloatArray:
    prices: List[float] = fm.guide("Prices with regex", regex=r"\d+\.\d+")


@fm.generable("Invalid: minimum guide on array of floats")
class InvalidMinimumOnFloatArray:
    prices: List[float] = fm.guide("Prices with minimum", minimum=0.0)


@fm.generable("Invalid: maximum guide on array of floats")
class InvalidMaximumOnFloatArray:
    prices: List[float] = fm.guide("Prices with maximum", maximum=1000.0)


@fm.generable("Invalid: range guide on array of floats")
class InvalidRangeOnFloatArray:
    prices: List[float] = fm.guide("Prices with range", range=(0.0, 1000.0))


@fm.generable("Invalid: anyOf guide on array of strings")
class InvalidAnyOfOnStringArray:
    tags: List[str] = fm.guide("Tags with anyOf", anyOf=["tag1", "tag2"])


@fm.generable("Invalid: regex guide on array of strings")
class InvalidRegexOnStringArray:
    tags: List[str] = fm.guide("Tags with regex", regex=r"[a-z]+")


@fm.generable("Invalid: minimum guide on array of strings")
class InvalidMinimumOnStringArray:
    tags: List[str] = fm.guide("Tags with minimum", minimum=1)


@fm.generable("Invalid: maximum guide on array of strings")
class InvalidMaximumOnStringArray:
    tags: List[str] = fm.guide("Tags with maximum", maximum=10)


@fm.generable("Invalid: range guide on array of strings")
class InvalidRangeOnStringArray:
    tags: List[str] = fm.guide("Tags with range", range=(1, 10))


@pytest.mark.asyncio
async def test_unsupported_guide_numeric_on_string(model):
    """Test that numeric guides on string properties throw UnsupportedGuide error."""
    print("\n=== Testing UnsupportedGuide: Numeric Guides on String ===")

    session = fm.LanguageModelSession("You are a helpful assistant.", model=model)

    test_cases = [
        (InvalidNumericGuidesOnString, "minimum on string"),
        (InvalidMaximumOnString, "maximum on string"),
        (InvalidRangeOnString, "range on string"),
        (InvalidCountOnString, "count on string"),
        (InvalidMinItemsOnString, "minItems on string"),
        (InvalidMaxItemsOnString, "maxItems on string"),
    ]

    for test_class, description in test_cases:
        with pytest.raises(fm.UnsupportedGuideError):
            await session.respond(
                "Generate a test object.",
                generating=test_class,
            )
        print(f"✓ {description}: Correctly threw UnsupportedGuide error")


@pytest.mark.asyncio
async def test_unsupported_guide_string_on_numeric(model):
    """Test that string guides on numeric properties throw UnsupportedGuide error."""
    print("\n=== Testing UnsupportedGuide: String Guides on Numeric Types ===")

    session = fm.LanguageModelSession("You are a helpful assistant.", model=model)

    test_cases = [
        (InvalidAnyOfOnInteger, "anyOf on integer"),
        (InvalidRegexOnInteger, "regex on integer"),
        (InvalidCountOnInteger, "count on integer"),
        (InvalidMinItemsOnInteger, "minItems on integer"),
        (InvalidMaxItemsOnInteger, "maxItems on integer"),
        (InvalidAnyOfOnFloat, "anyOf on float"),
        (InvalidRegexOnFloat, "regex on float"),
        (InvalidCountOnFloat, "count on float"),
        (InvalidMinItemsOnFloat, "minItems on float"),
        (InvalidMaxItemsOnFloat, "maxItems on float"),
    ]

    for test_class, description in test_cases:
        with pytest.raises(fm.UnsupportedGuideError):
            await session.respond(
                "Generate a test object.",
                generating=test_class,
            )
        print(f"✓ {description}: Correctly threw UnsupportedGuide error")


@pytest.mark.asyncio
async def test_unsupported_guide_on_arrays(model):
    """Test that invalid guides on array properties throw UnsupportedGuide error."""
    print("\n=== Testing UnsupportedGuide: Invalid Guides on Arrays ===")

    session = fm.LanguageModelSession("You are a helpful assistant.", model=model)

    test_cases = [
        # Array of integers
        (InvalidAnyOfOnIntArray, "anyOf on array<integer>"),
        (InvalidRegexOnIntArray, "regex on array<integer>"),
        (InvalidMinimumOnIntArray, "minimum on array<integer>"),
        (InvalidMaximumOnIntArray, "maximum on array<integer>"),
        (InvalidRangeOnIntArray, "range on array<integer>"),
        # Array of floats
        (InvalidAnyOfOnFloatArray, "anyOf on array<float>"),
        (InvalidRegexOnFloatArray, "regex on array<float>"),
        (InvalidMinimumOnFloatArray, "minimum on array<float>"),
        (InvalidMaximumOnFloatArray, "maximum on array<float>"),
        (InvalidRangeOnFloatArray, "range on array<float>"),
        # Array of strings - these guides don't apply to the array itself
        # anyOf *does* work on array<string>, so it's not included here
        (InvalidRegexOnStringArray, "regex on array<string>"),
        (InvalidMinimumOnStringArray, "minimum on array<string>"),
        (InvalidMaximumOnStringArray, "maximum on array<string>"),
        (InvalidRangeOnStringArray, "range on array<string>"),
    ]

    for test_class, description in test_cases:
        with pytest.raises(fm.UnsupportedGuideError):
            print("Testing:", description)
            await session.respond(
                "Generate a test object.",
                generating=test_class,
            )
        print(f"✓ {description}: Correctly threw UnsupportedGuide error")
