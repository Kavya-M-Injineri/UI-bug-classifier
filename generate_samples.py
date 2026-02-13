"""
Generate sample UI bug screenshots for training.
Creates synthetic images for each bug category so you can test the training pipeline.
"""

import os
from PIL import Image, ImageDraw, ImageFont

DATASET_DIR = "dataset"
SAMPLES_PER_CLASS = 30  # Minimum viable for testing

COLORS = {
    "bg": (245, 245, 250),
    "header": (100, 80, 200),
    "text": (30, 30, 60),
    "card": (255, 255, 255),
    "border": (200, 200, 220),
    "accent": (139, 92, 246),
    "error": (220, 50, 50),
    "green": (34, 197, 94),
}


def draw_base_ui(draw, w, h):
    """Draw a basic UI layout."""
    # Header bar
    draw.rectangle([0, 0, w, 50], fill=COLORS["header"])
    draw.text((15, 15), "MyApp Dashboard", fill=(255, 255, 255))

    # Sidebar
    draw.rectangle([0, 50, 180, h], fill=(240, 238, 255))
    for i, label in enumerate(["Home", "Profile", "Settings", "Help"]):
        y = 70 + i * 40
        draw.rectangle([10, y, 170, y + 30], fill=COLORS["card"], outline=COLORS["border"])
        draw.text((20, y + 8), label, fill=COLORS["text"])


def generate_layout_broken(idx):
    """Generate images with broken layout patterns."""
    import random
    random.seed(idx)
    img = Image.new("RGB", (224, 224), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    draw_base_ui(draw, 224, 224)

    # Simulate overlapping / misaligned elements
    draw.rectangle([180, 50, 224, 224], fill=COLORS["bg"])
    # Random overlapping boxes
    for _ in range(3):
        x = random.randint(60, 200)
        y = random.randint(60, 200)
        draw.rectangle([x, y, x + random.randint(50, 150), y + random.randint(30, 80)],
                       fill=(random.randint(200, 255), random.randint(200, 255), 255),
                       outline=COLORS["error"], width=2)
    # Broken grid line
    draw.line([0, random.randint(80, 180), 224, random.randint(80, 180)], fill=COLORS["error"], width=3)
    return img


def generate_text_overflow(idx):
    """Generate images with text overflow patterns."""
    import random
    random.seed(idx + 100)
    img = Image.new("RGB", (224, 224), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    draw_base_ui(draw, 224, 224)

    # Long overflowing text
    overflow_text = "ThisIsAVeryLongTextStringThatOverflowsBeyondTheContainer" * 2
    y_pos = random.randint(70, 150)
    draw.rectangle([190, y_pos, 220, y_pos + 40], fill=COLORS["card"], outline=COLORS["border"])
    draw.text((195, y_pos + 5), overflow_text, fill=COLORS["text"])

    # Text running outside box
    draw.rectangle([190, y_pos + 50, 215, y_pos + 80], fill=COLORS["card"], outline=COLORS["border"])
    draw.text((192, y_pos + 55), "Overflowing content that goes beyond boundaries", fill=COLORS["error"])
    return img


def generate_dark_mode_issue(idx):
    """Generate images with dark mode contrast problems."""
    import random
    random.seed(idx + 200)
    img = Image.new("RGB", (224, 224), (30, 30, 40))  # Dark background
    draw = ImageDraw.Draw(img)

    # Dark header on dark bg (low contrast)
    draw.rectangle([0, 0, 224, 50], fill=(40, 40, 50))
    draw.text((15, 15), "MyApp Dashboard", fill=(60, 60, 70))  # Nearly invisible

    # Dark cards on dark bg
    for i in range(3):
        y = 60 + i * 55
        draw.rectangle([15, y, 210, y + 45], fill=(35, 35, 45), outline=(40, 40, 50))
        draw.text((25, y + 15), f"Card {i+1} - Low contrast text", fill=(50, 50, 60))

    # One element with white text (inconsistent)
    draw.rectangle([15, 230 - 55, 210, 230 - 10], fill=(35, 35, 45))
    draw.text((25, 230 - 45), "This text IS visible", fill=(255, 255, 255))
    return img


def generate_alignment_issue(idx):
    """Generate images with alignment problems."""
    import random
    random.seed(idx + 300)
    img = Image.new("RGB", (224, 224), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    draw_base_ui(draw, 224, 224)

    # Misaligned cards
    offsets = [random.randint(-15, 15) for _ in range(3)]
    for i in range(3):
        x = 190 + offsets[i]
        y = 60 + i * 55
        draw.rectangle([x, y, x + 30, y + 40], fill=COLORS["card"], outline=COLORS["accent"], width=2)
        draw.text((x + 5, y + 12), f"Item", fill=COLORS["text"])

    # Uneven spacing indicator
    for i in range(4):
        x = 190 + random.randint(-10, 10)
        draw.ellipse([x, 60 + i * 40, x + 8, 68 + i * 40], fill=COLORS["error"])
    return img


def generate_no_bug(idx):
    """Generate clean, well-aligned UI screenshots."""
    import random
    random.seed(idx + 400)
    img = Image.new("RGB", (224, 224), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    draw_base_ui(draw, 224, 224)

    # Clean content area
    for i in range(3):
        y = 60 + i * 55
        draw.rectangle([190, y, 218, y + 45], fill=COLORS["card"], outline=COLORS["border"])
        draw.text((195, y + 5), f"Title {i+1}", fill=COLORS["text"])
        # Progress bar
        bar_w = random.randint(10, 20)
        draw.rectangle([195, y + 28, 195 + bar_w, y + 34], fill=COLORS["green"])
        draw.rectangle([195 + bar_w, y + 28, 215, y + 34], fill=COLORS["border"])
    return img


GENERATORS = {
    "Layout Broken": generate_layout_broken,
    "Text Overflow": generate_text_overflow,
    "Dark Mode Issue": generate_dark_mode_issue,
    "Alignment Issue": generate_alignment_issue,
    "No Bug": generate_no_bug,
}


def main():
    print("🖼️  Generating sample UI bug screenshots...")
    print(f"   {SAMPLES_PER_CLASS} images per class × {len(GENERATORS)} classes")
    print()

    total = 0
    for class_name, generator in GENERATORS.items():
        class_dir = os.path.join(DATASET_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(SAMPLES_PER_CLASS):
            img = generator(i)
            img.save(os.path.join(class_dir, f"sample_{i:03d}.png"))
            total += 1

        print(f"   ✅ {class_name}: {SAMPLES_PER_CLASS} images")

    print(f"\n🎉 Generated {total} total images in '{DATASET_DIR}/'")
    print(f"   Now run: python train_model.py")


if __name__ == "__main__":
    main()
