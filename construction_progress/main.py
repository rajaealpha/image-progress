"""
CLI entry point.

Usage examples:
  # Analyse a single image
  python -m construction_progress.main --image path/to/photo.jpg

  # Specify camera ID and deployment name
  python -m construction_progress.main --image photo.jpg --camera cam_north --deployment my-gpt4o

  # Watch a folder and process new images automatically
  python -m construction_progress.main --watch /photos/cam_01 --camera cam_01 --interval 300

  # Print current progress summary from history
  python -m construction_progress.main --status --camera cam_01
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

# Force UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _resolve_image(image_path: str) -> tuple[str, bool]:
    """
    If image_path is an HTTP/HTTPS URL, download it to a temp file.
    Returns (local_path, is_temp).  Caller must delete temp file if is_temp=True.
    Uses httpx with SSL verification disabled to handle corporate/Azure Blob certs.
    """
    if image_path.startswith("http://") or image_path.startswith("https://"):
        import httpx
        logger.info("Downloading image from URL...")
        suffix = ".jpg"
        url_path = image_path.split("?")[0]
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            if url_path.lower().endswith(ext):
                suffix = ext
                break
        tf = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tf.close()
        try:
            with httpx.Client(verify=False, timeout=60, follow_redirects=True) as client:
                response = client.get(image_path)
                response.raise_for_status()
                with open(tf.name, "wb") as f:
                    f.write(response.content)
            logger.info("Downloaded %d KB to %s", len(response.content) // 1024, tf.name)
            return tf.name, True
        except Exception as e:
            os.unlink(tf.name)
            raise RuntimeError(f"Failed to download image from URL: {e}") from e
    return image_path, False


def run_single(image_path: str, camera_id: str, deployment: str) -> int:
    from construction_progress.pipeline.orchestrator import ConstructionProgressPipeline

    try:
        local_path, is_temp = _resolve_image(image_path)
    except RuntimeError as e:
        logger.error("%s", e)
        return 1

    try:
        if not Path(local_path).exists():
            logger.error("Image not found: %s", local_path)
            return 1

        with ConstructionProgressPipeline(camera_id=camera_id, deployment_name=deployment) as pipeline:
            result = pipeline.run(local_path)
    finally:
        if is_temp and os.path.exists(local_path):
            os.unlink(local_path)

    if not result.success:
        logger.error("Pipeline failed: %s", result.error)
        return 1

    r = result.report
    print("\n" + "=" * 60)
    print(f"  Construction Progress Report")
    print(f"  Timestamp : {r.run_timestamp}")
    print(f"  Camera    : {camera_id}")
    print(f"  Overall   : {r.overall_progress_pct:.1f}%")
    print("=" * 60)
    for z in r.zone_scores:
        review = " [REVIEW]" if z.needs_human_review else ""
        stalled = " [STALLED]" if z.stalled else ""
        delta = f" (+{z.delta_pct:.1f}%)" if z.delta_pct > 0 else (f" ({z.delta_pct:.1f}%)" if z.delta_pct < 0 else "")
        print(f"  {z.zone_name:<20} {z.progress_pct:5.1f}%{delta}  [{z.stage_label}]{review}{stalled}")
    print("=" * 60)
    if r.alerts:
        print("\n  ALERTS:")
        for a in r.alerts:
            print(f"    [!] {a}")
    print(f"\n  Dashboard : {result.output_paths.get('dashboard', 'N/A')}")
    print(f"  Report    : {result.output_paths.get('json_report', 'N/A')}")
    print(f"  Annotated : {result.output_paths.get('annotated_image', 'N/A')}")
    print()
    return 0


def watch_folder(folder: str, camera_id: str, deployment: str, interval: int):
    from construction_progress.pipeline.orchestrator import ConstructionProgressPipeline

    folder_path = Path(folder)
    if not folder_path.exists():
        logger.error("Watch folder not found: %s", folder)
        sys.exit(1)

    processed = set()
    logger.info("Watching %s every %ds...", folder, interval)

    with ConstructionProgressPipeline(camera_id=camera_id, deployment_name=deployment) as pipeline:
        while True:
            images = sorted(folder_path.glob("*.jpg")) + sorted(folder_path.glob("*.png"))
            new_images = [p for p in images if str(p) not in processed]
            for img in new_images:
                logger.info("New image detected: %s", img.name)
                result = pipeline.run(str(img))
                if result.success:
                    logger.info("Processed %s → %.1f%%", img.name, result.report.overall_progress_pct)
                else:
                    logger.error("Failed %s: %s", img.name, result.error)
                processed.add(str(img))
            time.sleep(interval)


def print_status(camera_id: str):
    from construction_progress.storage.history_store import HistoryStore
    store = HistoryStore()
    snap = store.get_latest_snapshot(camera_id)
    if snap is None:
        print(f"No history found for camera '{camera_id}'.")
        return

    print(f"\nLatest snapshot for '{camera_id}': {snap['timestamp']}")
    print(f"Overall progress: {snap['overall_progress']:.1f}%")
    for zid, score in snap.get("zone_scores", {}).items():
        print(f"  {score.get('zone_name', zid):<20} {score.get('progress_pct', 0):5.1f}%  [{score.get('stage_label', '')}]")

    timeline = store.get_progress_timeline(camera_id)
    print(f"\nHistory: {len(timeline)} snapshots recorded.")


def main():
    parser = argparse.ArgumentParser(
        description="Construction site progress monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image",      help="Path to input image for single analysis")
    parser.add_argument("--watch",      help="Folder to watch for new images")
    parser.add_argument("--status",     action="store_true", help="Print current status from history")
    parser.add_argument("--camera",     default="cam_01", help="Camera ID (default: cam_01)")
    parser.add_argument("--deployment", default=None,    help="Azure deployment name override")
    parser.add_argument("--interval",   type=int, default=300, help="Watch interval in seconds (default: 300)")
    parser.add_argument("--verbose",    action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.status:
        print_status(args.camera)
    elif args.image:
        sys.exit(run_single(args.image, args.camera, args.deployment))
    elif args.watch:
        watch_folder(args.watch, args.camera, args.deployment, args.interval)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
