# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Test Foundation Models image input support in prompts.

Tests cover:
- Text & images combined in a prompt
- Only an image with instructions in the session
- Guided generation with images
- Image attachments with labels
"""

from pathlib import Path
from typing import List
import apple_fm_sdk as fm
import pytest


# Get the path to test resources
TEST_RESOURCES_DIR = Path(__file__).parent / "resources"
SIMPLE_IMAGE = TEST_RESOURCES_DIR / "test-simple-image.jpeg"
TEXT_DENSE_IMAGE = TEST_RESOURCES_DIR / "test-text-dense-image.png"


@pytest.mark.asyncio
async def test_text_and_image_prompt(session):
    """Test prompt with both text and image."""
    print("\n=== Testing Text + Image Prompt ===")

    try:
        # Create a prompt with both text and image
        image = fm.ImageAttachment(path=SIMPLE_IMAGE)
        prompt = [
            "What do you see in this image? Describe it briefly.",
            image,
        ]

        response = await session.respond(prompt)
        assert isinstance(response, str), f"Invalid response type: {type(response)}"
        assert len(response) > 0, "Response should not be empty"
        print(f"✓ Got response: {response[:100]}...")
    except fm.ImagePromptError as e:
        pytest.skip(f"Image prompts not supported: {e}")


@pytest.mark.asyncio
async def test_multiple_images_with_text(session):
    """Test prompt with multiple images and text."""
    print("\n=== Testing Multiple Images + Text Prompt ===")

    try:
        # Create a prompt with text and multiple images
        image1 = fm.ImageAttachment(path=SIMPLE_IMAGE)
        image2 = fm.ImageAttachment(path=TEXT_DENSE_IMAGE)
        prompt = [
            "I'm going to show you two images.",
            image1,
            "and here's another one:",
            image2,
            "What do you see in these images?",
        ]

        response = await session.respond(prompt)
        assert isinstance(response, str), f"Invalid response type: {type(response)}"
        assert len(response) > 0, "Response should not be empty"
        print(f"✓ Got response: {response[:100]}...")
    except fm.ImagePromptError as e:
        pytest.skip(f"Image prompts not supported: {e}")


@pytest.mark.asyncio
async def test_image_only_with_session_instructions(model):
    """Test prompt with only an image, using session instructions."""
    print("\n=== Testing Image-Only Prompt with Session Instructions ===")

    # Create session with instructions
    session = fm.LanguageModelSession(
        instructions="You are a helpful assistant that describes images in detail.",
        model=model,
    )

    try:
        # Create a prompt with only an image
        image = fm.ImageAttachment(path=SIMPLE_IMAGE)
        prompt = image

        response = await session.respond(prompt)
        assert isinstance(response, str), f"Invalid response type: {type(response)}"
        assert len(response) > 0, "Response should not be empty"
        print(f"✓ Got response: {response[:100]}...")
    except fm.ImagePromptError as e:
        pytest.skip(f"Image prompts not supported: {e}")


@pytest.mark.asyncio
async def test_image_list_with_session_instructions(model):
    """Test prompt with a list containing only images, using session instructions."""
    print("\n=== Testing Image List Prompt with Session Instructions ===")

    # Create session with instructions
    session = fm.LanguageModelSession(
        instructions="Compare the images and explain their differences.",
        model=model,
    )

    try:
        # Create a prompt with only images in a list
        image1 = fm.ImageAttachment(path=SIMPLE_IMAGE)
        image2 = fm.ImageAttachment(path=TEXT_DENSE_IMAGE)
        prompt: list[fm.PromptComponent] = [image1, image2]

        response = await session.respond(prompt)
        assert isinstance(response, str), f"Invalid response type: {type(response)}"
        assert len(response) > 0, "Response should not be empty"
        print(f"✓ Got response: {response[:100]}...")
    except fm.ImagePromptError as e:
        pytest.skip(f"Image prompts not supported: {e}")


# Define a structured response schema for image analysis
@fm.generable("Image analysis")
class ImageAnalysis:
    description: str = fm.guide("Description of what is visible in the image")
    main_elements: List[str] = fm.guide(
        "Main elements or objects visible in the image", count=3
    )
    colors: List[str] = fm.guide("Prominent colors in the image", count=2)


@pytest.mark.asyncio
async def test_guided_generation_with_image(model):
    """Test guided generation with an image prompt."""
    print("\n=== Testing Guided Generation with Image ===")

    session = fm.LanguageModelSession(
        instructions="You are a helpful image analyst.",
        model=model,
    )

    try:
        # Create a prompt with text and image for guided generation
        image = fm.ImageAttachment(path=SIMPLE_IMAGE)
        prompt = [
            "Analyze this image:",
            image,
        ]

        result = await session.respond(prompt, generating=ImageAnalysis)
        assert type(result) is ImageAnalysis, (
            f"Invalid generated content type: {type(result)}"
        )

        print(f"✓ Got structured result: {type(result).__name__}")
        print(f"  Description: {result.description}")
        print(f"  Main elements: {result.main_elements}")
        print(f"  Colors: {result.colors}")

        # Validate the structured results
        assert isinstance(result.description, str) and len(result.description) > 0
        assert len(result.main_elements) == 3
        assert len(result.colors) == 2
    except fm.ImagePromptError as e:
        pytest.skip(f"Image prompts not supported: {e}")


@pytest.mark.asyncio
async def test_guided_generation_with_schema_and_image(session):
    """Test guided generation with schema parameter and image."""
    print("\n=== Testing Guided Generation with Schema and Image ===")

    try:
        # Get the schema from the generable class
        schema = ImageAnalysis.generation_schema()

        # Create a prompt with text and image
        image = fm.ImageAttachment(path=TEXT_DENSE_IMAGE)
        prompt = [
            "Analyze this image:",
            image,
        ]

        generated_content = await session.respond(prompt, schema=schema)
        assert isinstance(generated_content, fm.GeneratedContent), (
            f"Invalid generated content type: {type(generated_content)}"
        )

        # Extract values from the generated content
        description = generated_content.value(str, for_property="description")
        main_elements = generated_content.value(List[str], for_property="main_elements")
        colors = generated_content.value(List[str], for_property="colors")

        print("✓ Got structured result using schema")
        print(f"  Description: {description}")
        print(f"  Main elements: {main_elements}")
        print(f"  Colors: {colors}")

        # Validate constraints
        assert isinstance(description, str) and len(description) > 0
        assert len(main_elements) == 3
        assert len(colors) == 2
    except fm.ImagePromptError as e:
        pytest.skip(f"Image prompts not supported: {e}")


@pytest.mark.asyncio
async def test_guided_generation_with_multiple_images(model):
    """Test guided generation with multiple images."""
    print("\n=== Testing Guided Generation with Multiple Images ===")

    # Define a comparison schema
    @fm.generable("Image comparison")
    class ImageComparison:
        first_image_description: str = fm.guide("Description of the first image")
        second_image_description: str = fm.guide("Description of the second image")
        similarities: List[str] = fm.guide("Similarities between images", count=2)
        differences: List[str] = fm.guide("Differences between images", count=2)

    session = fm.LanguageModelSession(
        instructions="Compare the images provided.",
        model=model,
    )

    try:
        # Create a prompt with multiple images
        image1 = fm.ImageAttachment(path=SIMPLE_IMAGE)
        image2 = fm.ImageAttachment(path=TEXT_DENSE_IMAGE)
        prompt = [
            "First image:",
            image1,
            "Second image:",
            image2,
            "Compare these two images.",
        ]

        result = await session.respond(prompt, generating=ImageComparison)
        assert type(result) is ImageComparison, (
            f"Invalid generated content type: {type(result)}"
        )

        print("✓ Got comparison result")
        print(f"  First image: {result.first_image_description}")
        print(f"  Second image: {result.second_image_description}")
        print(f"  Similarities: {result.similarities}")
        print(f"  Differences: {result.differences}")

        # Validate the structured results
        assert (
            isinstance(result.first_image_description, str)
            and len(result.first_image_description) > 0
        )
        assert (
            isinstance(result.second_image_description, str)
            and len(result.second_image_description) > 0
        )
        assert len(result.similarities) == 2
        assert len(result.differences) == 2
    except fm.ImagePromptError as e:
        pytest.skip(f"Image prompts not supported: {e}")


@pytest.mark.asyncio
async def test_image_prompt_error_handling(session):
    """Test that invalid image paths are handled properly."""
    print("\n=== Testing Image Error Handling ===")

    # Create a prompt with non-existent image
    try:
        image = fm.ImageAttachment(path=Path("/nonexistent/image.png"))
        prompt = [
            "What's in this image?",
            image,
        ]

        # Should raise an error for invalid image path
        with pytest.raises((fm.ImagePromptError, fm.FoundationModelsError)):
            await session.respond(prompt)
    except fm.ImagePromptError:
        # ImageAttachment constructor itself may raise the error
        print("✓ ImageAttachment correctly raised error for non-existent file")


@pytest.mark.asyncio
async def test_attachment_with_label(session):
    """Test prompt with ImageAttachment using an explicit label."""
    print("\n=== Testing ImageAttachment with Label ===")

    try:
        image = fm.ImageAttachment(path=SIMPLE_IMAGE, label="photo1")
        prompt = [
            "What do you see in the image labeled photo1?",
            image,
        ]

        response = await session.respond(prompt)
        assert isinstance(response, str), f"Invalid response type: {type(response)}"
        assert len(response) > 0, "Response should not be empty"
        print(f"✓ Got response: {response[:100]}...")
    except fm.ImagePromptError as e:
        pytest.skip(f"Image prompts not supported: {e}")


@pytest.mark.asyncio
async def test_attachment_without_label(session):
    """Test prompt with ImageAttachment without a label."""
    print("\n=== Testing ImageAttachment without Label ===")

    try:
        image = fm.ImageAttachment(path=SIMPLE_IMAGE)
        prompt = [
            "Describe what you see in this image.",
            image,
        ]

        response = await session.respond(prompt)
        assert isinstance(response, str), f"Invalid response type: {type(response)}"
        assert len(response) > 0, "Response should not be empty"
        print(f"✓ Got response: {response[:100]}...")
    except fm.ImagePromptError as e:
        pytest.skip(f"Image prompts not supported: {e}")


@pytest.mark.asyncio
async def test_multiple_attachments_with_labels(session):
    """Test prompt with multiple ImageAttachment components with different labels."""
    print("\n=== Testing Multiple ImageAttachments with Labels ===")

    try:
        image1 = fm.ImageAttachment(path=SIMPLE_IMAGE, label="image-a")
        image2 = fm.ImageAttachment(path=TEXT_DENSE_IMAGE, label="image-b")
        prompt = [
            "I'm going to show you two labeled images.",
            image1,
            image2,
            "What do you see in image-a and image-b?",
        ]

        response = await session.respond(prompt)
        assert isinstance(response, str), f"Invalid response type: {type(response)}"
        assert len(response) > 0, "Response should not be empty"
        print(f"✓ Got response: {response[:100]}...")
    except fm.ImagePromptError as e:
        pytest.skip(f"Image prompts not supported: {e}")


@pytest.mark.asyncio
async def test_text_dense_image_analysis(session):
    """Test analyzing an image with dense text content."""
    print("\n=== Testing Text-Dense Image Analysis ===")

    try:
        image = fm.ImageAttachment(path=TEXT_DENSE_IMAGE)
        prompt = [
            "What text or information can you see in this image?",
            image,
        ]

        response = await session.respond(prompt)
        assert isinstance(response, str), f"Invalid response type: {type(response)}"
        assert len(response) > 0, "Response should not be empty"
        print(f"✓ Got response: {response[:100]}...")
    except fm.ImagePromptError as e:
        pytest.skip(f"Image prompts not supported: {e}")
