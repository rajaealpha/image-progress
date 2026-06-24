HOW TO ADD REFERENCE IMAGES
============================

Folder structure:
  reference_images/
    <zone_id>/
      0.jpg        ← what the zone looks like at 0% (empty / site cleared)
      10.jpg       ← 10% complete
      20.jpg       ← 20% complete
      ...
      100.jpg      ← 100% complete (fully finished)

Rules:
  - Folder name must match the zone "id" in config.py (e.g. "overall_yard")
  - File name must be the percentage number: 0.jpg, 10.jpg, 25.jpg, 50.jpg etc.
  - You don't need every 10% — just paste whatever milestones you have
    e.g. 0.jpg, 30.jpg, 60.jpg, 100.jpg is fine
  - Supported formats: .jpg .jpeg .png
  - Images should be taken from the same camera angle as your live feed

Current zones configured:
  - overall_yard  → put images in: reference_images/overall_yard/

When you add more zones to config.py, create matching subfolders here.
