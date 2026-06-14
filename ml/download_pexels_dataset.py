# SPDX-FileCopyrightText: 2026 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
AutoSolve Pexels Dataset Downloader.
Downloads standard CC-licensed tracking reference clips from Pexels API
and categorizes them into standard categories with correct filename format.
"""

import os
import sys
import json
import time
import urllib.request
import urllib.parse
import cv2

# Map of categories and Pexels search queries
CATEGORIES = {
    'indoor': ['room walk', 'interior tracking', 'museum walk'],
    'outdoor': ['city walk', 'street tracking', 'mountain pan'],
    'drone': ['drone landscape', 'aerial flyover city'],
    'handheld': ['handheld follow', 'running camera'],
    'gimbal': ['cinematic dolly', 'gimbal walk'],
    'action': ['action skate', 'fast run camera']
}

def get_fps_and_resolution(filepath):
    """
    Open the video using OpenCV to get its exact FPS and resolution.
    Returns: (fps_str, res_str)
    """
    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return "30fps", "1080p"
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Format FPS
        if fps > 0:
            fps_rounded = round(fps)
            fps_str = f"{fps_rounded}fps"
        else:
            fps_str = "30fps"
            
        # Format resolution and ratio
        if (width == 1920 and height == 1080) or (width == 1080 and height == 1920):
            res_str = "1080p" if width > height else "1080p_vertical"
        elif (width == 1280 and height == 720) or (width == 720 and height == 1280):
            res_str = "720p" if width > height else "720p_vertical"
        elif width >= 3840 or height >= 2160:
            res_str = "4k"
        elif width == height:
            res_str = f"{width}x{height}_square"
        else:
            res_str = f"{height}p" if width > height else f"{width}p_vertical"
            
        return fps_str, res_str
    except Exception as e:
        print(f"Warning: Failed to parse video metadata with OpenCV: {e}")
        return "30fps", "1080p"

def download_file(url, output_path):
    """Downloads a file showing progress."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            block_size = 8192
            
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    sys.stdout.write(f"\rDownloading... {percent:.1f}% ({downloaded}/{total_size} bytes)")
                    sys.stdout.flush()
            print() # new line
        return True
    except Exception as e:
        print(f"\nError downloading {url}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def query_pexels_videos(query, api_key, limit=5):
    """Queries Pexels Video Search API."""
    params = {
        'query': query,
        'per_page': limit
    }
    url = f"https://api.pexels.com/videos/search?{urllib.parse.urlencode(params)}"
    
    req = urllib.request.Request(url)
    req.add_header('Authorization', api_key)
    req.add_header('User-Agent', 'AutoSolve Dataset Generator')
    
    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                return json.loads(response.read().decode('utf-8'))
            else:
                print(f"Pexels API error status code: {response.status}")
                return None
    except Exception as e:
        print(f"Error querying Pexels API: {e}")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download reference tracking dataset from Pexels API")
    parser.add_argument("--api-key", required=True, help="Pexels API Key")
    parser.add_argument("--clips-dir", default="ml/clips", help="Target directory for downloaded clips")
    parser.add_argument("--clips-per-cat", type=int, default=2, help="Number of clips to download per category")
    parser.add_argument("--max-duration", type=int, default=20, help="Max video duration in seconds to filter out long clips")
    parser.add_argument("--min-duration", type=int, default=5, help="Min video duration in seconds")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.clips_dir):
        os.makedirs(args.clips_dir, exist_ok=True)
        
    print(f"Starting Pexels Dataset Downloader...")
    print(f"Target directory: {args.clips_dir}")
    print(f"Clips per category: {args.clips_per_cat}")
    print(f"Duration range: {args.min_duration}s - {args.max_duration}s")
    
    clip_id_counter = 1
    downloaded_clips = []
    seen_video_ids = set()
    
    for category, queries in CATEGORIES.items():
        print(f"\n==========================================")
        print(f"Processing Category: {category.upper()}")
        print(f"==========================================")
        
        category_downloaded_count = 0
        
        # Loop through search queries for this category
        for query in queries:
            if category_downloaded_count >= args.clips_per_cat:
                break
                
            print(f"Searching query: '{query}'...")
            # Query more than we need to allow filtering
            limit_val = max(20, args.clips_per_cat * 3)
            results = query_pexels_videos(query, args.api_key, limit=limit_val)
            if not results or "videos" not in results:
                print("No results returned or error occurred.")
                continue
                
            videos = results["videos"]
            print(f"Found {len(videos)} potential landscape videos.")
            
            for video in videos:
                if category_downloaded_count >= args.clips_per_cat:
                    break
                    
                video_id = video["id"]
                duration = video["duration"]
                
                # Prevent downloading duplicate videos
                if video_id in seen_video_ids:
                    continue
                
                # Filter by duration
                if duration < args.min_duration or duration > args.max_duration:
                    print(f"  Skipping video {video_id}: duration is {duration}s (outside range)")
                    continue
                    
                # Find best mp4 video file (preferring HD 1080p or 720p or any)
                video_files = video.get("video_files", [])
                selected_file = None
                
                # Look for HD 1080p (either landscape 1920x1080 or vertical 1080x1920)
                for vf in video_files:
                    if vf.get("file_type") == "video/mp4" and (vf.get("width") == 1920 or vf.get("height") == 1920):
                        selected_file = vf
                        break
                        
                # Fallback to HD 720p (either landscape 1280x720 or vertical 720x1280)
                if not selected_file:
                    for vf in video_files:
                        if vf.get("file_type") == "video/mp4" and (vf.get("width") == 1280 or vf.get("height") == 1280):
                            selected_file = vf
                            break
                            
                # Fallback to any mp4 file
                if not selected_file:
                    for vf in video_files:
                        if vf.get("file_type") == "video/mp4":
                            selected_file = vf
                            break
                            
                if not selected_file:
                    print(f"  Skipping video {video_id}: no suitable mp4 file found")
                    continue
                    
                download_link = selected_file["link"]
                width = selected_file.get("width", 1920)
                height = selected_file.get("height", 1080)
                
                print(f"  Selected video {video_id}: {width}x{height}, duration: {duration}s")
                
                # Temporary filename for OpenCV parsing
                temp_filename = os.path.join(args.clips_dir, f"temp_{video_id}.mp4")
                
                print(f"  Downloading video to temp file...")
                success = download_file(download_link, temp_filename)
                if not success:
                    continue
                    
                # Use OpenCV to determine exact FPS and resolution
                fps_str, res_str = get_fps_and_resolution(temp_filename)
                
                # Format final filename: clip_[id]_[category]_[fps]_[resolution].mp4
                final_name = f"clip_{clip_id_counter:03d}_{category}_{fps_str}_{res_str}.mp4"
                final_path = os.path.join(args.clips_dir, final_name)
                
                # Rename temp file to final file name
                try:
                    if os.path.exists(final_path):
                        os.remove(final_path)
                    os.rename(temp_filename, final_path)
                    print(f"  -> Successfully saved as: {final_name}")
                    
                    downloaded_clips.append({
                        "id": clip_id_counter,
                        "category": category,
                        "fps": fps_str,
                        "resolution": res_str,
                        "filename": final_name,
                        "pexels_id": video_id,
                        "duration": duration,
                        "user": video.get("user", {}).get("name", "Unknown")
                    })
                    
                    clip_id_counter += 1
                    category_downloaded_count += 1
                    seen_video_ids.add(video_id)
                    
                    # Be nice to the Pexels API / rate limiting
                    time.sleep(1.0)
                except Exception as e:
                    print(f"  Error renaming temp file to final path: {e}")
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
                        
        print(f"Downloaded {category_downloaded_count} clips for category '{category}'.")
        
    print("\n==========================================")
    print("Download Summary:")
    print("==========================================")
    print(f"Total downloaded clips: {len(downloaded_clips)}")
    for clip in downloaded_clips:
        creator_name = clip.get('user', 'Unknown')
        try:
            print(f"- [{clip['id']:03d}] {clip['filename']} (Duration: {clip['duration']}s, Creator: {creator_name})")
        except UnicodeEncodeError:
            creator_safe = creator_name.encode('ascii', 'replace').decode('ascii')
            print(f"- [{clip['id']:03d}] {clip['filename']} (Duration: {clip['duration']}s, Creator: {creator_safe})")
        
    # Write a summary metadata JSON file in clips dir
    summary_path = os.path.join(args.clips_dir, "dataset_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as sf:
        json.dump(downloaded_clips, sf, indent=4)
    print(f"Dataset summary JSON saved to '{summary_path}'")

if __name__ == "__main__":
    main()
