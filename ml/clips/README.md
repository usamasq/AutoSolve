# AutoSolve Reference Video Dataset Library

This directory serves as the library index for the 100+ reference video clips used to benchmark AutoSolve and collect dataset samples for machine learning training. All clips should be CC-licensed or royalty-free, representing diverse real-world camera tracking scenarios.

---

## Clip Library Schema

For training, collect videos in standard container formats (e.g. `.mp4`, `.mov`) at different frame rates (24fps, 30fps, 60fps) and resolutions (1080p, 4K). 

Videos are categorized into 7 core motion and environmental classes:

### 1. Indoor (20 clips)
* **Description**: Controlled lighting, high texture on objects (carpet, wood, bricks), minimal reflections, and close range parallax.
* **Sources**: Pexels, Pixabay (search: "interior tracking", "room walk", "museum camera movement").
* **Key Challenges**: Short distance focus, lens distortion at wide angles.

### 2. Outdoor (20 clips)
* **Description**: Natural sunlight, long-range landscape depth, foliage, buildings, streets.
* **Sources**: Pexels, Pixabay (search: "city walk", "street tracking", "mountain pan").
* **Key Challenges**: Moving trees/leaves, variable lighting, high sky ratio (needs sky exclusion).

### 3. Drone / Aerial (15 clips)
* **Description**: Large height altitude, clear vertical parallax, terrain details.
* **Sources**: Pixabay (search: "drone landscape", "aerial flyover city").
* **Key Challenges**: Near-zero ground parallax if flying too high, massive sky regions at horizons, fast rotation.

### 4. Handheld (15 clips)
* **Description**: Visible jitter, rapid changes in acceleration, fast tilt/pan shifts.
* **Sources**: Custom recorded or Pexels (search: "handheld follow", "running camera").
* **Key Challenges**: Motion blur, erratic search area shifts, sudden out-of-frame tracking.

### 5. Gimbal / Stabilized (10 clips)
* **Description**: Extremely smooth linear translation, dolly push-in, truck left/right.
* **Sources**: Pexels (search: "cinematic dolly", "gimbal walk").
* **Key Challenges**: Low local acceleration makes track drift harder to identify without global coherence.

### 6. Action / Fast Motion (10 clips)
* **Description**: Extreme motion blur, fast panning, action sports.
* **Sources**: Action footage archives, Pexels (search: "action skate", "fast run camera").
* **Key Challenges**: Tracking markers through heavy blur, requiring large search windows and robust motion models.

### 7. VFX Plates (10 clips)
* **Description**: Footage shot specifically for camera tracking and integration. Often includes physical tracking markers (crosses, tennis balls) and zoom/crane shifts.
* **Sources**: Hollywood Camera Work tracking plates, VFX library samples.
* **Key Challenges**: Extracting track coordinates precisely, calibrating focal lengths.

---

## Dataset Storage

Raw clips should be placed in `ml/clips/` using the naming pattern `clip_[id]_[category]_[fps]_[resolution].mp4` (e.g. `clip_001_indoor_24fps_1080p.mp4`).

Run the dataset generator runner:
```bash
python ml/run_collection.py --clips-dir ml/clips --output-dir ml/data/raw --blender-bin "path/to/blender"
```
This runs the background tracking simulations and populates `ml/data/raw/` with JSON samples.
