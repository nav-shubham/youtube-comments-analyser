import re
import time
import logging
from datetime import datetime
import yt_dlp
import database

logger = logging.getLogger(__name__)

def clean_handle_or_url(input_str):
    """
    Parses a channel URL, custom handle, or ID to a standard format yt-dlp can use.
    """
    input_str = input_str.strip()
    
    # If it's a full URL
    if "youtube.com/" in input_str or "youtu.be/" in input_str:
        return input_str
        
    # If it is a handle like @AnuandDiv
    if input_str.startswith("@"):
        return f"https://www.youtube.com/{input_str}"
        
    # If it looks like a direct channel ID (starts with UC)
    if len(input_str) == 24 and input_str.startswith("UC"):
        return f"https://www.youtube.com/channel/{input_str}"
        
    # Default to handle search
    return f"https://www.youtube.com/@{input_str}"

def get_channel_videos(channel_input, limit=None, cookies_browser=None):
    """
    Fetches all videos for a channel using yt-dlp.
    Returns:
        channel_id (str): Resolved channel ID.
        channel_title (str): Resolved channel title.
        videos (list): List of parsed video dictionaries.
    """
    cleaned_input = clean_handle_or_url(channel_input)
    
    logger.info(f"Resolving channel using: {cleaned_input}")
    
    # 1. Extract flat channel page first to get Channel ID and Title
    ydl_opts_flat = {
        'extract_flat': True,
        'playlistend': 5,
        'quiet': True,
        'skip_download': True,
    }
    
    if cookies_browser and cookies_browser != "None":
        ydl_opts_flat['cookiesfrombrowser'] = (cookies_browser.lower(),)
    
    channel_id = None
    channel_title = "Unknown Channel"
    
    with yt_dlp.YoutubeDL(ydl_opts_flat) as ydl:
        try:
            info = ydl.extract_info(cleaned_input, download=False)
            channel_title = info.get('title') or "Unknown Channel"
            info_id = info.get('id')
            
            # If the extracted ID is the channel ID (starts with UC)
            if info_id and info_id.startswith("UC"):
                channel_id = info_id
            else:
                # Check entries, which represent tabs (Videos, Live, Shorts, etc.)
                entries = info.get('entries', [])
                for entry in entries:
                    entry_id = entry.get('id')
                    if entry_id and entry_id.startswith("UC"):
                        channel_id = entry_id
                        break
        except Exception as e:
            logger.error(f"Error resolving channel flat info: {str(e)}")
            
    # Fallback if channel ID couldn't be resolved
    if not channel_id:
        if "channel/" in cleaned_input:
            match = re.search(r"channel/(UC[a-zA-Z0-9_-]{22})", cleaned_input)
            if match:
                channel_id = match.group(1)
        if not channel_id:
            # Extract handle/name from URL
            match = re.search(r"youtube\.com/(?:@|c/)?([^/]+)", cleaned_input)
            channel_id = match.group(1) if match else "unknown_channel"
            
    if channel_title == "Unknown Channel":
        channel_title = channel_id
        
    logger.info(f"Resolved Channel ID: {channel_id}, Title: {channel_title}")
    
    # 2. Construct the UU Uploads playlist URL
    if channel_id.startswith("UC"):
        uploads_playlist_id = "UU" + channel_id[2:]
        playlist_url = f"https://www.youtube.com/playlist?list={uploads_playlist_id}"
    else:
        # Fallback to appending /videos
        if cleaned_input.endswith("/videos"):
            playlist_url = cleaned_input
        else:
            playlist_url = cleaned_input.rstrip("/") + "/videos"
            
    logger.info(f"Fetching video entries from uploads index: {playlist_url}")
    
    # 3. Extract all video entries from the uploads playlist
    ydl_opts_playlist = {
        'extract_flat': True,
        'playlistend': limit,
        'quiet': True,
        'skip_download': True,
    }
    
    if cookies_browser and cookies_browser != "None":
        ydl_opts_playlist['cookiesfrombrowser'] = (cookies_browser.lower(),)
    
    parsed_videos = []
    with yt_dlp.YoutubeDL(ydl_opts_playlist) as ydl:
        try:
            playlist_info = ydl.extract_info(playlist_url, download=False)
            
            # Refine channel title if it says "Uploads from ..."
            p_title = playlist_info.get('title')
            if p_title and "Uploads from" in p_title:
                channel_title = p_title.replace("Uploads from ", "").strip()
                
            entries = playlist_info.get('entries', [])
            for entry in entries:
                video_id = entry.get('id')
                if video_id:
                    # Clean view count formatting
                    view_count = "0 views"
                    views_val = entry.get('view_count')
                    if views_val is not None:
                        try:
                            view_count = f"{int(views_val):,} views"
                        except ValueError:
                            view_count = f"{views_val} views"
                            
                    # Clean upload date formatting (YYYYMMDD to YYYY-MM-DD)
                    upload_date = "Unknown"
                    date_val = entry.get('upload_date')
                    if date_val and len(date_val) == 8:
                        upload_date = f"{date_val[:4]}-{date_val[4:6]}-{date_val[6:]}"
                    elif entry.get('upload_date'):
                        upload_date = entry.get('upload_date')
                        
                    parsed_videos.append({
                        'video_id': video_id,
                        'title': entry.get('title') or "Untitled",
                        'published_time': upload_date,
                        'view_count': view_count
                    })
        except Exception as e:
            logger.error(f"Error fetching playlist details: {str(e)}")
            
    return channel_id, channel_title, parsed_videos

def scrape_channel_comments(channel_input, video_limit=None, comment_limit_per_video=100, 
                            cookies_browser=None, progress_callback=None, cancel_check=None):
    """
    Orchestrates the entire scraping job:
    1. Fetches channel videos using yt-dlp.
    2. Saves channel & videos to database.
    3. Loops through each video, fetches comments via yt-dlp, and bulk saves to SQLite database.
    
    This downloader is a pure network scraper and contains zero analytical logic.
    """
    if progress_callback:
        progress_callback(0, 100, "Resolving channel and fetching video list...")
        
    channel_id, channel_title, videos = get_channel_videos(channel_input, limit=video_limit, cookies_browser=cookies_browser)
    total_videos = len(videos)
    
    if total_videos == 0:
        if progress_callback:
            progress_callback(100, 100, "No videos found for this channel. Check URL or toggle browser cookies.")
        return channel_id, channel_title, 0
        
    logger.info(f"Found {total_videos} videos. Saving channel info...")
    
    # Save channel metadata
    # Guess custom handle
    handle = ""
    if "@" in channel_input:
        match = re.search(r"(@[a-zA-Z0-9_.-]+)", channel_input)
        if match:
            handle = match.group(1)
            
    database.save_channel(channel_id, channel_title, handle, channel_input)
    
    # Save videos metadata
    for vid in videos:
        database.save_video(vid['video_id'], channel_id, vid['title'], vid['published_time'], vid['view_count'])
        
    total_comments_saved = 0
    
    # Loop videos and extract comments
    for i, vid in enumerate(videos):
        if cancel_check and cancel_check():
            logger.info("Scraping job cancelled by user.")
            if progress_callback:
                progress_callback(int(100 * (i / total_videos)), 100, f"Scraping cancelled. Saved {total_comments_saved} comments.")
            break
            
        video_id = vid['video_id']
        video_title = vid['title']
        
        percent_complete = int(100 * (i / total_videos))
        if progress_callback:
            progress_callback(
                percent_complete, 
                100, 
                f"Processing video {i+1}/{total_videos}: \"{video_title[:30]}...\" (Total saved: {total_comments_saved})"
            )
            
        logger.info(f"Scraping comments for video {i+1}/{total_videos}: {video_id}")
        
        # Configure yt-dlp to download only comments
        ydl_opts_comments = {
            'getcomments': True,
            'skip_download': True,
            'quiet': True,
            'extractor_args': {
                'youtube': {
                    'max_comments': [str(comment_limit_per_video)]
                }
            }
        }
        
        if cookies_browser and cookies_browser != "None":
            ydl_opts_comments['cookiesfrombrowser'] = (cookies_browser.lower(),)
            
        comments_list = []
        try:
            with yt_dlp.YoutubeDL(ydl_opts_comments) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                comments_list = info.get('comments', [])
        except Exception as e:
            logger.error(f"Error scraping comments for video {video_id} with yt-dlp: {str(e)}")
            if progress_callback:
                progress_callback(
                    percent_complete, 
                    100, 
                    f"⚠️ Bot block/error for video {i+1}. Skipping video."
                )
                time.sleep(1.0)
            continue
            
        comments_batch = []
        
        for comment_item in comments_list:
            if cancel_check and cancel_check():
                break
                
            comment_text = comment_item.get('text', '')
            
            # Format published time from timestamp
            timestamp_val = comment_item.get('timestamp')
            published_time = "Unknown"
            if timestamp_val:
                try:
                    published_time = datetime.fromtimestamp(timestamp_val).strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    published_time = comment_item.get('time_text', '')
            else:
                published_time = comment_item.get('time_text', '')
                
            comments_batch.append({
                'comment_id': comment_item.get('id'),
                'video_id': video_id,
                'author': comment_item.get('author') or "Anonymous",
                'author_channel': comment_item.get('author_id') or "",
                'text': comment_text,
                'likes': comment_item.get('like_count', 0) or 0,
                'published_time': published_time,
                'heart': 1 if comment_item.get('author_is_uploader', False) else 0,
                'reply': 1 if comment_item.get('parent') and comment_item.get('parent') != 'root' else 0,
                # Sentiment calculations removed from download layer!
                'sentiment_score': 0.0,
                'sentiment_label': 'Neutral'
            })
            
            # Flush batch to SQLite in sets of 500
            if len(comments_batch) >= 500:
                database.save_comments(comments_batch)
                total_comments_saved += len(comments_batch)
                comments_batch = []
                
        # Save any remaining comments in the final batch
        if comments_batch:
            database.save_comments(comments_batch)
            total_comments_saved += len(comments_batch)
            
        # Pacing sleep to avoid heavy rate limits from YouTube
        time.sleep(0.5)
        
    if progress_callback:
        progress_callback(100, 100, f"Completed! Scraped {total_comments_saved} comments across {total_videos} videos.")
        
    return channel_id, channel_title, total_comments_saved
