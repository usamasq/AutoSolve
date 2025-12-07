# AutoSolve - One-Click Camera Tracking for Blender

AutoSolve is a Blender extension that provides automated camera tracking using Structure-from-Motion (SfM). It replaces Blender's manual tracking workflow with a single "Analyze & Solve" button.

## Features

- **One-Click Solve** - Automatic feature detection, matching, and camera motion solving
- **GPU Point Cloud** - Real-time visualization of solved 3D points (100k+ at 60 FPS)
- **Ground Wand** - Click to align the scene to world coordinates
- **Scale Tool** - Define real-world scale with two clicks
- **Deep Integration** - Camera backgrounds, compositor undistortion, render settings

## Requirements

- Blender 4.2.0 or later
- pycolmap (bundled with extension)

## Installation

1. Download the extension from the Blender Extensions Platform
2. In Blender, go to Edit → Preferences → Add-ons
3. Click "Install from Disk" and select the downloaded file
4. Enable the extension

## Usage

1. Open the **VFX Workspace** in Blender
2. In the Movie Clip Editor, load your footage
3. In the sidebar (N), find the **AutoSolve** tab
4. Click **Analyze & Solve**
5. Use **Ground Wand** to align the scene
6. Click **Create Scene Objects** to finalize

## Platform Support

| Platform                    | Status           |
| --------------------------- | ---------------- |
| Windows x64                 | ✅ Supported     |
| macOS ARM64 (Apple Silicon) | ✅ Supported     |
| Linux x64                   | ✅ Supported     |
| macOS Intel                 | ❌ Not supported |

## License

GPL-3.0-or-later
