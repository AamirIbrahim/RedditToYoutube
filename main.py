import os
import requests
import re
from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips
import praw
from pydub import AudioSegment
import numpy as np

import subprocess
import math
import textwrap

from PIL import Image, ImageDraw, ImageFont
import math
import textwrap

import pickle
import google.auth
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from googleapiclient.http import MediaFileUpload
from dotenv import load_dotenv
import json

SUBREDDIT_COUNTS_FILE = "subreddit_counts.json"
# Load environment variables from .env file
load_dotenv()

client_secrets_file = 'client_secret.json'
credentials_file = 'youtube_credentials.pickle'

# Define the scopes required for the YouTube Data API
scopes = ["https://www.googleapis.com/auth/youtube.upload", 'https://www.googleapis.com/auth/youtube.readonly']

def create_text_image(text, font_path, font_size, color, outline_color, max_width, position="center"):
    # Create a temporary image to calculate text size
    temp_img = Image.new('RGBA', (1920, 1080), (255, 255, 255, 0))
    draw = ImageDraw.Draw(temp_img)
    font = ImageFont.truetype(font_path, font_size)
    
    # Wrap text to fit within max_width
    lines = textwrap.wrap(text, width=30)
    wrapped_text = "\n".join(lines)
    
    # Calculate text size
    text_size = draw.multiline_textsize(wrapped_text, font=font)
    img_height = text_size[1] + 20  # Add some padding
    img = Image.new('RGBA', (1920, 1080), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Calculate text position based on the specified position
    if position == "top-left":
        text_position = (10, 10)  # Small margin from the top-left corner
    elif position == "bottom-right":
        text_position = (1920 - text_size[0] - 10, 1080 - text_size[1] - 10)  # Small margin from the bottom-right corner
    elif position == "center":
        text_position = ((1920 - text_size[0]) // 2, (1080 - text_size[1]) // 2)  # Centered
    else:
        raise ValueError("Invalid position. Use 'top-left', 'bottom-right', or 'center'.")
    
    # Draw the outline by drawing the text multiple times with slight offsets
    outline_range = range(-1, 2)
    for x in outline_range:
        for y in outline_range:
            draw.multiline_text((text_position[0] + x, text_position[1] + y), wrapped_text, font=font, fill=outline_color)
    
    # Draw the text on the image
    draw.multiline_text(text_position, wrapped_text, font=font, fill=color)
    
    return img

def add_text_overlay(input_video, title, author, output_video):
    """Adds a text overlay to a video."""
    # Create the title text image
    font_path = "font.TTF"
    font_size = 50
    color = (255, 255, 255)  # White
    outline_color = (0, 0, 0)  # Black
    max_width = 1920

    # Estimate reading time for the title
    words = len(title.split())
    reading_speed_wpm = 200  # Average reading speed in words per minute
    reading_time = words / reading_speed_wpm * 60  # Convert to seconds
    title_duration = min(reading_time, input_video.duration)  # Use the shorter duration

    title_img = create_text_image(title, font_path, font_size, color, outline_color, max_width, position="top-left")
    title_img_np = np.array(title_img)  # Convert PIL image to NumPy array
    title_clip = ImageClip(title_img_np, duration=title_duration).set_position(("left", "top"))

    # Create the author text image
    author_img = create_text_image("/u/" + str(author), font_path, font_size, color, outline_color, max_width, position="bottom-right")
    author_img_np = np.array(author_img)  # Convert PIL image to NumPy array
    author_clip = ImageClip(author_img_np, duration=title_duration).set_position(("right", "bottom"))

    # Composite the video with the text clips and logo
    final_video = CompositeVideoClip([input_video, title_clip, author_clip])
    return final_video

def is_audio_silent(audio_path, silence_threshold=-40.0):
    """Checks if an audio file is silent (all sound below threshold)."""
    audio = AudioSegment.from_file(audio_path, "mp4")
    samples = np.array(audio.get_array_of_samples())  # Convert to NumPy array

    # Convert to dBFS (volume level)
    volume = audio.dBFS if len(samples) > 0 else -100.0  

    print(f"Audio volume (dB): {volume}")  # Debugging: Show detected volume

    return volume < silence_threshold  # True if silent, False if audible

def get_audio_url(video_url, bitrate="128"):
    """Constructs the correct DASH audio URL for a Reddit video"""
    import re
    match = re.search(r"https://v\.redd\.it/([\w\d]+)/", video_url)
    if match:
        post_id = match.group(1)
        return f"https://v.redd.it/{post_id}/DASH_AUDIO_{bitrate}.mp4"
    return None

def find_best_audio(video_url):
    """Finds the best available audio file (128kbps preferred, fallback to 64kbps)"""
    for bitrate in ["128", "64"]:  # Try 128 first, then 64
        audio_url = get_audio_url(video_url, bitrate)
        response = requests.head(audio_url)
        return audio_url if response.status_code == 200 else None
        
def check_audio_exists(audio_url):
    """Checks if the audio file exists by making a HEAD request"""
    
def download_file(url, filename):
    """Downloads a file from a given URL with headers to avoid 403 errors."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "https://www.reddit.com/"
    }
    
    response = requests.get(url, headers=headers, stream=True)
    
    if response.status_code == 403:
        print(f"Access forbidden: {url}")
        return False  # Return False if access is denied
    
    response.raise_for_status()  # Raise error for bad responses
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    
    return True  # Return True if download succeeds

def has_audio(video_path):
    """Checks if a video file already contains an audio track"""
    video = VideoFileClip(video_path)
    return video.audio is not None
    
def normalize_audio(audio_path, output_path):
    """Boosts the volume of an audio file if it's too quiet."""
    audio = AudioSegment.from_file(audio_path)
    louder_audio = audio.apply_gain(10)  # Increase volume by 10 dB
    louder_audio.export(output_path, format="mp4")

def resize_video_to_1080p_exact(video_path):
    """Resizes a video to exactly 1920x1080."""
    clip = VideoFileClip(video_path)
    return clip.resize(newsize=(1920, 1080))  # Resize to 1920x1080 (forced)

def concatenate_clips(clips):
    """Concatenates a list of video file paths into a single video."""
    video_clips = [VideoFileClip(clip) for clip in clips]  # Load each file as a VideoFileClip
    final_clip = concatenate_videoclips(video_clips)  # Concatenate the VideoFileClip objects
    final_clip.write_videofile('final_video.mp4', codec='libx264', threads=16, fps=60)

def authenticate_youtube():
    """Authenticate and return the YouTube service."""
    creds = None
    # Load credentials from file if they exist
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no valid credentials, authenticate the user
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for future use
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return build('youtube', 'v3', credentials=creds)

def upload_video(youtube, video_file, title, description, tags, category_id, privacy_status):
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": category_id
        },
        "status": {
            "selfDeclaredMadeForKids": False,
            "madeForKids": False,
            "privacyStatus": privacy_status
        }
    }
    
    media_body = MediaFileUpload(video_file, chunksize=-1, resumable=True)

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media_body
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Uploaded {int(status.progress() * 100)}%")

    print("Upload Complete!")
    print("Video URL: https://www.youtube.com/watch?v=" + response['id'])

def resize_video_to_1080p_exact(video_path):
    """Resizes a video to exactly 1920x1080 while maintaining aspect ratio."""
    clip = VideoFileClip(video_path)
    # Calculate the aspect ratio of the video
    video_aspect_ratio = clip.w / clip.h
    target_aspect_ratio = 1920 / 1080

    if video_aspect_ratio > target_aspect_ratio:
        # Video is wider than 16:9, fit width and add vertical padding
        clip = clip.resize(width=1920)
        new_height = int(1920 / video_aspect_ratio)
        clip = clip.margin(top=(1080 - new_height) // 2, bottom=(1080 - new_height) // 2, color=(0, 0, 0))
    else:
        # Video is taller than 16:9, fit height and add horizontal padding
        clip = clip.resize(height=1080)
        new_width = int(1080 * video_aspect_ratio)
        clip = clip.margin(left=(1920 - new_width) // 2, right=(1920 - new_width) // 2, color=(0, 0, 0))

    return clip

def load_subreddit_counts():
    """Load subreddit counts from the JSON file."""
    if os.path.exists(SUBREDDIT_COUNTS_FILE):
        try:
            with open(SUBREDDIT_COUNTS_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Warning: {SUBREDDIT_COUNTS_FILE} is empty or corrupted. Resetting file.")
            return {}
    return {}

def save_subreddit_counts(counts):
    """Save subreddit counts to the JSON file."""
    with open(SUBREDDIT_COUNTS_FILE, "w") as f:
        json.dump(counts, f, indent=4)

def get_video_count_for_subreddit(subreddit):
    """Get and increment the video count for a specific subreddit."""
    counts = load_subreddit_counts()
    if subreddit not in counts:
        counts[subreddit] = 0
    counts[subreddit] += 1
    save_subreddit_counts(counts)
    return counts[subreddit]

def get_authenticated_service():
    credentials = None

    # Check if credentials are already stored
    if os.path.exists(credentials_file):
        with open(credentials_file, 'rb') as token:
            credentials = pickle.load(token)

    # If there are no (valid) credentials available, let the user log in.
    if not credentials:
        flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(client_secrets_file, scopes)
        credentials = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open(credentials_file, 'wb') as token:
            pickle.dump(credentials, token)
    youtube = googleapiclient.discovery.build("youtube", "v3", credentials=credentials)

    return youtube

def main():
    # Reddit API credentials
    reddit = praw.Reddit(client_id = os.getenv('CLIENT_ID'), \
                        client_secret = os.getenv('CLIENT_SECRET'), \
                        user_agent = os.getenv('USER_AGENT'))

    # Subreddit to scrape
    subreddit = reddit.subreddit(os.getenv('SUBREDDIT'))
    
    # Directory to save videos
    os.makedirs("input", exist_ok=True)
    os.makedirs("upload", exist_ok=True)
    
    broadcasters = []
    duration_video = []
    duration = 0
    formatted_clips = []
    reddit_links = []
    # Fetch top posts of the day
    for submission in subreddit.top(time_filter="week"):  # Top posts of the week
        if submission.is_video:
            video_url = submission.media['reddit_video']['fallback_url']
            
            # File paths
            video_filename = f"input/{submission.id}_video.mp4"
            audio_filename = f"input/{submission.id}_audio.mp4"
            merged_filename = f"upload/{submission.id}_merged.mp4"

            # Download the video
            download_file(video_url, video_filename)
            
            # Check if an audio file exists
            audio_url = find_best_audio(video_url)
            
            if audio_url and submission.media['reddit_video']['has_audio']:
                print(f"Downloading audio: {audio_url}")
                download_file(audio_url, audio_filename)
                if is_audio_silent(audio_filename):
                    print("⚠️ Warning: Audio file is silent. Skipping merge.")
                else:
                    print("🔊 Audio detected, normalizing volume...")
                    normalized_audio = "normalized_audio.mp4"
                    normalize_audio(audio_filename, normalized_audio)

                    video = VideoFileClip(video_filename)
                    
                    # Let's get the reddit link to put in the description
                    reddit_link = f"https://www.reddit.com{submission.permalink}"
                    reddit_links.append(reddit_link)

                    minutes, seconds = divmod(duration, 60)
                    duration_video.append("%02d:%02d" % (minutes, seconds))
                    duration = duration + video.duration

                    video = resize_video_to_1080p_exact(video_filename)
                    video = add_text_overlay(video, submission.title, submission.author, merged_filename)

                    audio = AudioFileClip(normalized_audio)
                    video = video.set_audio(audio)
                    
                    video.write_videofile(merged_filename, codec="libx264", audio_codec="aac")
                    formatted_clips.append(merged_filename)

                    # Break the loop if total duration exceeds 10 minutes (600 seconds)
                    if duration >= 600:
                        print("✅ Total duration exceeds 10 minutes. Stopping.")
                        break

    # Put all of our clips together, and then upload
    concatenate_clips(formatted_clips)
    
    video_file = "final_video.mp4"

    youtube = get_authenticated_service()

    video_count = get_video_count_for_subreddit(str(subreddit))
    
    title = "Best of r/{0} Week #{1}".format(subreddit, video_count)
    description = "Featured Reddit Links: \n{0}".format("\n".join("{} {}".format(x, y) for x,y in zip(duration_video, reddit_links)))
    tags = broadcasters
    category_id = "24" # Entertainment
    privacy_status = "public"
    
    upload_video(youtube, video_file, title, description, tags, category_id, privacy_status)

if __name__ == '__main__':
    main()