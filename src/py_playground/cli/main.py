"""Main CLI entry point for PyPlayground."""

import argparse
import sys
import os
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="pyplayground",
        description="PyPlayground - AI & Data Science Playground",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pyplayground demo              Run the pose visualization demo
  pyplayground video             Generate a video animation
  pyplayground info              Show package information
        """
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Demo command
    demo_parser = subparsers.add_parser(
        "demo",
        help="Run the pose visualization demo"
    )
    demo_parser.add_argument(
        "--save",
        type=str,
        metavar="PATH",
        help="Save the output video to PATH"
    )
    demo_parser.add_argument(
        "--frames",
        type=int,
        default=60,
        help="Number of frames (default: 60)"
    )
    demo_parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)"
    )
    
    # Video command
    video_parser = subparsers.add_parser(
        "video",
        help="Generate a video animation"
    )
    video_parser.add_argument(
        "-o", "--output",
        type=str,
        default="pose_rotation.mp4",
        help="Output video path (default: pose_rotation.mp4)"
    )
    video_parser.add_argument(
        "--frames",
        type=int,
        default=120,
        help="Number of frames (default: 120)"
    )
    video_parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)"
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show package information"
    )
    
    return parser


def run_demo(args) -> int:
    """Run the pose visualization demo."""
    from ..visualization.pose import PoseVisualizer
    
    print("Running pose visualization demo...")
    
    viz = PoseVisualizer()
    
    if args.save:
        print(f"Generating {args.frames} frames at {args.fps} fps...")
        viz.save_video(args.save, frames=args.frames, fps=args.fps)
        print(f"Video saved to: {args.save}")
    else:
        print("Displaying interactive visualization...")
        print("Close the window to exit.")
        viz.setup_plot()
        import numpy as np
        viz.draw_frame(
            rotation=(0.5, 0.3, 0.2),
            translation=np.array([1, 0.5, 0])
        )
        viz.show()
    
    return 0


def run_video(args) -> int:
    """Generate a video animation."""
    from ..visualization.pose import PoseVisualizer
    
    print(f"Generating video: {args.output}")
    print(f"  Frames: {args.frames}")
    print(f"  FPS: {args.fps}")
    
    viz = PoseVisualizer()
    viz.save_video(args.output, frames=args.frames, fps=args.fps)
    
    # Get absolute path
    abs_path = os.path.abspath(args.output)
    print(f"\nVideo saved to: {abs_path}")
    
    # Try to open the video
    if sys.platform == "darwin":
        import subprocess
        try:
            subprocess.run(["open", abs_path], check=True)
            print("Opened video with default player.")
        except Exception as e:
            print(f"Could not auto-open video: {e}")
    
    return 0


def run_info(args) -> int:
    """Show package information."""
    import py_playground
    
    print("=" * 50)
    print("PyPlayground")
    print("=" * 50)
    print(f"Version: {py_playground.__version__}")
    print(f"Author: {py_playground.__author__}")
    print()
    print("Available modules:")
    print("  - py_playground.visualization.pose")
    print()
    print("Quick start:")
    print("  pyplayground demo       # Run interactive demo")
    print("  pyplayground video      # Generate a video")
    print("=" * 50)
    
    return 0


def main(args: list = None) -> int:
    """Main entry point."""
    parser = create_parser()
    parsed = parser.parse_args(args)
    
    if parsed.command is None:
        parser.print_help()
        return 1
    
    commands = {
        "demo": run_demo,
        "video": run_video,
        "info": run_info,
    }
    
    handler = commands.get(parsed.command)
    if handler:
        return handler(parsed)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
