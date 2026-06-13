# AutoSolve ML Training Guide for Beginners 🚀

If you want to train your own custom ML models for AutoSolve without getting bogged down in complex command lines, follow this straightforward guide. It is designed for Google Colab/Google Drive usage.

---

## 📂 Step 1: Collect Your Footage
1. Gather a set of video clips (different resolutions, ratios, framerates, and camera movements).
2. Create a folder in your **Google Drive** named `AutoSolve_Footage`.
3. Upload all your video clips into that folder.

---

## 📓 Step 2: Open the Jupyter Notebook
1. Upload [AutoSolve_Training.ipynb](file:///c:/Users/usama/OneDrive/Desktop/AutoSolve/ml/AutoSolve_Training.ipynb) to your Google Drive.
2. Double-click it and select **"Open with Google Colab"** (or run it locally on your computer using Jupyter).
3. If using Google Colab, change your runtime type to use a **GPU** for much faster processing:
   * Go to `Runtime → Change runtime type`
   * Select **T4 GPU** (or any available GPU) and click **Save**.

---

## ⚡ Step 3: Run the Notebook (The "One-Click" Run)
The notebook is organized sequentially. You can just run the cells one by one:

1. **Mount Google Drive:**
   * Run the first cell to connect Google Colab to your Google Drive. Follow the pop-up instructions to grant permissions.
2. **Install Dependencies:**
   * Run the next cell to automatically download and install PyTorch, ONNX, and OpenCV libraries.
3. **Point to your Footage:**
   * In the configuration cell, make sure the path points to your Google Drive folder:
     ```python
     FOOTAGE_DIR = "/content/drive/MyDrive/AutoSolve_Footage"
     ```
4. **Run Feature Extraction:**
   * Run this cell to automatically scan your videos and calculate movement speeds, lens distortion curvature, zoom rates, and tracker trajectories.
5. **Train the Models:**
   * Run the training cells. The notebook will automatically train:
     * **Settings Model:** Predicts optimal pattern/search sizes.
     * **Track Predictor:** Predicts when a track is going to slip/fail.
     * **Patch Rigidity:** Flags non-rigid regions (like water or foliage).
6. **Export Models:**
   * Run the final cells to generate the compiled files ready for Blender.

---

## 🚚 Step 4: Download & Install the Trained Models
Once training finishes, the notebook will save your new models in your Google Drive or Colab file browser. Download the files and copy them to the Blender addon folders:

### 1. Copy the ONNX models to `autosolve/tracker/models/`
Download and copy these three files:
* 📥 `settings_model.onnx`
* 📥 `track_predictor.onnx`
* 📥 `patch_rigidity.onnx`

Into your local Blender addon directory:
📂 [autosolve/tracker/models/](file:///c:/Users/usama/OneDrive/Desktop/AutoSolve/autosolve/tracker/models/)

---

### 2. Copy the NumPy model to `autosolve/tracker/models/`
Download and copy:
* 📥 `track_predictor.json`

Into the same directory:
📂 [autosolve/tracker/models/](file:///c:/Users/usama/OneDrive/Desktop/AutoSolve/autosolve/tracker/models/)

---

### 3. Copy the Presets to `autosolve/tracker/presets/`
Download and copy:
* 📥 `region_weights.json`
* 📥 `defaults.json`

Into the presets directory:
📂 [autosolve/tracker/presets/](file:///c:/Users/usama/OneDrive/Desktop/AutoSolve/autosolve/tracker/presets/)

---

## 🎉 Step 5: Reload and Solve!
1. Open Blender.
2. Go to `Preferences → Add-ons` and search for **AutoSolve**.
3. Disable and re-enable it (or restart Blender) to load the new models.
4. AutoSolve will now use your custom-trained intelligence for all tracking!

---

## 🤝 Step 6: Collaborative Training (Combining Data from Multiple People)
If you are working with other developers/creators who are also training models, you can easily combine your datasets:

### The Recommended Way: Combine Raw Logs (Before Training)
Since training depends on raw JSON feature files, you can share and merge your files:
1. Every person runs their own **Feature Extraction** step (Step 3, cell 4) on their respective video clips.
2. Under the Colab files pane, go to `ml/data/raw/` and download the generated `.json` files.
3. Have everyone upload their raw `.json` files into a single shared Google Drive folder (e.g., `AutoSolve_Combined_Raw_Data`).
4. In the Jupyter Notebook, point the dataset directory to this shared folder:
   ```python
   RAW_DATA_DIR = "/content/drive/MyDrive/AutoSolve_Combined_Raw_Data"
   ```
5. Run the **Train** and **Export** cells. The notebook will automatically merge all raw logs together, normalize the combined dataset, and train a single model that benefits from everyone's data!
