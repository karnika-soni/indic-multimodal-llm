# Multimodal Translation Dataset


Stores image-text pairs for multimodal neural machine translation.


Format:

images/
    image_001.jpg


metadata.json

Example:

{
 "image": "image_001.jpg",
 "source_text": "...",
 "target_text": "..."
}


Pipeline:

Image
→ Detectron2 region features
→ multimodal mBART
→ translation generation
