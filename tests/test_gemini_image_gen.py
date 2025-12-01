"""Test script for Gemini image generation.

This script validates the Gemini 3 Pro Image API works correctly
before integrating into the subagent system.

Run with: python tests/test_gemini_image_gen.py
"""
import os
import base64
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from project root .env
# Use override=True to ensure .env values take precedence over existing env vars
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"

# Check if key exists before loading
existing_key = os.getenv('GEMINI_API_KEY')
if existing_key:
    print(f"WARNING: GEMINI_API_KEY already in environment (ends with ...{existing_key[-4:]})")

load_dotenv(env_path, override=True)

print(f"Loading .env from: {env_path}")
api_key = os.getenv('GEMINI_API_KEY')
if api_key:
    print(f"GEMINI_API_KEY after .env load: Yes (ends with ...{api_key[-4:]})")
else:
    print("GEMINI_API_KEY present: No")


def test_langchain_image_gen():
    """Test image generation via LangChain ChatGoogleGenerativeAI."""
    from langchain_google_genai import ChatGoogleGenerativeAI, Modality

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment")
        return False

    print("Initializing Gemini 3 Pro Image model...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-pro-image-preview",
        google_api_key=api_key,
    )

    prompt = "Generate a simple, minimalist logo for a coffee shop called 'Bean & Brew'. Use warm brown and cream colors."
    print(f"Generating image with prompt: {prompt[:50]}...")

    try:
        response = llm.invoke(
            [{"role": "user", "content": prompt}],
            response_modalities=[Modality.TEXT, Modality.IMAGE],
        )

        print(f"Response type: {type(response)}")
        print(f"Response content blocks: {len(response.content)}")

        # Find image block in response
        for i, block in enumerate(response.content):
            print(f"Block {i}: {type(block)}")
            if isinstance(block, str):
                print(f"  Text: {block[:100]}...")
            elif isinstance(block, dict):
                print(f"  Dict keys: {block.keys()}")
                if block.get("image_url"):
                    url = block["image_url"]["url"]
                    print(f"  Image URL prefix: {url[:50]}...")

                    # Extract base64 data
                    base64_data = url.split(",")[-1]
                    image_bytes = base64.b64decode(base64_data)

                    # Save to file
                    output_dir = Path("tests/output")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / "test_coffee_logo.png"
                    output_path.write_bytes(image_bytes)

                    print(f"\nSUCCESS: Image saved to {output_path}")
                    print(f"Image size: {len(image_bytes)} bytes")
                    return True

        print("WARNING: No image block found in response")
        return False

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_native_sdk_image_gen():
    """Test image generation via native Google GenAI SDK."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("google-genai package not installed, skipping native SDK test")
        return None

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment")
        return False

    print("\nTesting native Google GenAI SDK...")
    client = genai.Client(api_key=api_key)

    prompt = "A simple icon of a steaming coffee cup, flat design, minimalist"
    print(f"Generating image with prompt: {prompt[:50]}...")

    try:
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )

        print(f"Response candidates: {len(response.candidates)}")

        for part in response.parts:
            if part.text is not None:
                print(f"Text response: {part.text[:100]}...")
            elif part.inline_data is not None:
                image = part.as_image()
                output_dir = Path("tests/output")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / "test_coffee_icon_native.png"
                image.save(str(output_path))
                print(f"\nSUCCESS: Image saved to {output_path}")
                return True

        print("WARNING: No image found in response")
        return False

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Gemini 3 Pro Image Generation")
    print("=" * 60)

    print("\n--- Test 1: LangChain Integration ---")
    langchain_result = test_langchain_image_gen()

    print("\n--- Test 2: Native SDK ---")
    native_result = test_native_sdk_image_gen()

    print("\n" + "=" * 60)
    print("Results:")
    print(f"  LangChain: {'PASS' if langchain_result else 'FAIL'}")
    print(f"  Native SDK: {'PASS' if native_result else 'FAIL' if native_result is False else 'SKIPPED'}")
    print("=" * 60)
