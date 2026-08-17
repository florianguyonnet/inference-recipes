#!/usr/bin/env python3
"""
Vision smoke test for the SGLang Qwen3.8-27B server.

Renders a synthetic image (a number on a noisy background), sends it through
the OpenAI chat completions API as base64, and checks the model reads it.
The vision tower is served through SGLang's Qwen3-VL path — this verifies it
is actually live, not just loaded.

Usage:
    python scripts/test_vision.py --url http://localhost:30000 --model Qwen3.8-27B-NVFP4
"""

import argparse
import base64
import io
import json
import random
import sys
import urllib.request


def make_image(digit: int) -> str:
    """Render the digit as a large white number on dark noise; return base64 PNG."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        sys.exit("PIL is required (installed with sglang[all]).")

    random.seed(7)
    img = Image.new("RGB", (512, 512))
    px = img.load()
    for y in range(512):
        for x in range(512):
            v = random.randint(0, 60)
            px[x, y] = (v, v, v // 2)
    draw = ImageDraw.Draw(img)
    # Big white digit, centered. Default bitmap font scaled up via a temp canvas.
    font_img = Image.new("RGB", (60, 100), (0, 0, 0))
    ImageDraw.Draw(font_img).text((8, 8), str(digit), fill=(255, 255, 255))
    font_img = font_img.resize((240, 400))
    img.paste(font_img, (136, 56))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:30000")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    digit = 7
    b64 = make_image(digit)

    body = {
        "model": args.model,
        "max_tokens": 512,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What single digit is shown in this image? Reply with just the digit.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
    }

    req = urllib.request.Request(
        f"{args.url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())

    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    print(f"Model answer: {content.strip()[:400]}")

    # qwen3 reasoning parser may put the visible answer after the thinking block.
    if str(digit) in content:
        print(f"OK: digit {digit} recognized — vision tower is live.")
    else:
        print(f"FAIL: expected digit {digit} in the answer.")
        sys.exit(1)


if __name__ == "__main__":
    main()
