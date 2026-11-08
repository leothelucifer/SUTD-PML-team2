"""Package rendered slide PNGs and speaker notes into a 16:9 PowerPoint file.

    python docs/presentation/tools/build_pptx.py <render-dir> <output.pptx>
"""

import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu

render, output = Path(sys.argv[1]), Path(sys.argv[2])
deck = json.loads((Path(__file__).resolve().parents[1] / "source" / "deck.json").read_text())
notes = json.loads((render / "notes.json").read_text())
images = sorted(render.glob("slide-*.png"))
if len(images) != len(deck["order"]):
    raise SystemExit(f"expected {len(deck['order'])} slides, found {len(images)} images")

prs = Presentation()
prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)  # 13.333 x 7.5 in
prs.core_properties.title = deck["title"]
prs.core_properties.author = "SUTD PML Team 2"
blank = prs.slide_layouts[6]
for slide_id, image, note in zip(deck["order"], images, notes):
    slide = prs.slides.add_slide(blank)
    picture = slide.shapes.add_picture(str(image), 0, 0, prs.slide_width, prs.slide_height)
    picture.name = f"slide {slide_id}"
    if note.strip():
        slide.notes_slide.notes_text_frame.text = note.strip()
output.parent.mkdir(parents=True, exist_ok=True)
prs.save(output)
print(f"Wrote {output} with {len(images)} slides")
