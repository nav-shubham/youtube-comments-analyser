import sqlite3
import os
import pandas as pd
from datetime import datetime

DB_NAME = "youtube_analysis.db"

def get_connection():
    """Returns a connection to the SQLite database with Row factory enabled."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    """Initializes the database schema if it doesn't already exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create Channels table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS channels (
        channel_id TEXT PRIMARY KEY,
        title TEXT,
        handle TEXT,
        custom_url TEXT,
        scraped_at TEXT
    )
    """)
    
    # Create Videos table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS videos (
        video_id TEXT PRIMARY KEY,
        channel_id TEXT,
        title TEXT,
        published_time TEXT,
        view_count TEXT,
        scraped_at TEXT,
        FOREIGN KEY (channel_id) REFERENCES channels (channel_id) ON DELETE CASCADE
    )
    """)
    
    # Create Comments table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS comments (
        comment_id TEXT PRIMARY KEY,
        video_id TEXT,
        author TEXT,
        author_channel TEXT,
        text TEXT,
        likes INTEGER DEFAULT 0,
        published_time TEXT,
        heart INTEGER DEFAULT 0, -- 0 = False, 1 = True
        reply INTEGER DEFAULT 0,  -- 0 = False, 1 = True
        sentiment_score REAL,
        sentiment_label TEXT,
        scraped_at TEXT,
        FOREIGN KEY (video_id) REFERENCES videos (video_id) ON DELETE CASCADE
    )
    """)
    
    conn.commit()
    conn.close()

def save_channel(channel_id, title, handle, custom_url):
    """Saves or updates a channel record."""
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("""
    INSERT OR REPLACE INTO channels (channel_id, title, handle, custom_url, scraped_at)
    VALUES (?, ?, ?, ?, ?)
    """, (channel_id, title, handle, custom_url, now))
    conn.commit()
    conn.close()

def save_video(video_id, channel_id, title, published_time, view_count):
    """Saves or updates a single video record."""
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("""
    INSERT OR REPLACE INTO videos (video_id, channel_id, title, published_time, view_count, scraped_at)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (video_id, channel_id, title, published_time, view_count, now))
    conn.commit()
    conn.close()

def save_comments(comments_list):
    """Bulk inserts or updates comment records."""
    if not comments_list:
        return
    
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    
    # Prepare tuples for executemany
    data = []
    for c in comments_list:
        data.append((
            c['comment_id'],
            c['video_id'],
            c['author'],
            c.get('author_channel', ''),
            c['text'],
            c.get('likes', 0),
            c.get('published_time', ''),
            1 if c.get('heart', False) else 0,
            1 if c.get('reply', False) else 0,
            c.get('sentiment_score', 0.0),
            c.get('sentiment_label', 'Neutral'),
            now
        ))
        
    cursor.executemany("""
    INSERT OR REPLACE INTO comments (
        comment_id, video_id, author, author_channel, text, likes, 
        published_time, heart, reply, sentiment_score, sentiment_label, scraped_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)
    
    conn.commit()
    conn.close()

def get_channels():
    """Retrieves all channels that have been scraped and stored."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM channels ORDER BY scraped_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_channel_summary(channel_id):
    """Gathers overall statistics for a given channel."""
    conn = get_connection()
    cursor = conn.cursor()
    
    stats = {}
    
    # Total Videos
    cursor.execute("SELECT COUNT(*) FROM videos WHERE channel_id = ?", (channel_id,))
    stats['total_videos'] = cursor.fetchone()[0]
    
    # Total Comments
    cursor.execute("""
    SELECT COUNT(*) FROM comments c 
    JOIN videos v ON c.video_id = v.video_id 
    WHERE v.channel_id = ?
    """, (channel_id,))
    stats['total_comments'] = cursor.fetchone()[0]
    
    # Sentiment Breakdown
    cursor.execute("""
    SELECT sentiment_label, COUNT(*) as count FROM comments c
    JOIN videos v ON c.video_id = v.video_id
    WHERE v.channel_id = ?
    GROUP BY sentiment_label
    """, (channel_id,))
    sentiment_rows = cursor.fetchall()
    stats['sentiment'] = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    for row in sentiment_rows:
        stats['sentiment'][row['sentiment_label']] = row['count']
        
    # Total Likes on comments
    cursor.execute("""
    SELECT SUM(c.likes) FROM comments c
    JOIN videos v ON c.video_id = v.video_id
    WHERE v.channel_id = ?
    """, (channel_id,))
    stats['total_comment_likes'] = cursor.fetchone()[0] or 0
    
    # Heart Count
    cursor.execute("""
    SELECT COUNT(*) FROM comments c
    JOIN videos v ON c.video_id = v.video_id
    WHERE v.channel_id = ? AND c.heart = 1
    """, (channel_id,))
    stats['total_hearts'] = cursor.fetchone()[0]
    
    conn.close()
    return stats

def get_comments_dataframe(channel_id=None, video_id=None, search_query=None, sentiment_label=None, min_likes=0):
    """
    Queries comments database and returns a Pandas DataFrame for analysis.
    Supports complex filtering.
    """
    conn = get_connection()
    
    query = """
    SELECT 
        c.comment_id, c.video_id, v.title as video_title, c.author, c.text, 
        c.likes, c.published_time, c.heart, c.reply, c.sentiment_score, c.sentiment_label
    FROM comments c
    JOIN videos v ON c.video_id = v.video_id
    WHERE 1=1
    """
    params = []
    
    if channel_id:
        query += " AND v.channel_id = ?"
        params.append(channel_id)
        
    if video_id:
        query += " AND c.video_id = ?"
        params.append(video_id)
        
    if search_query:
        query += " AND c.text LIKE ?"
        params.append(f"%{search_query}%")
        
    if sentiment_label:
        query += " AND c.sentiment_label = ?"
        params.append(sentiment_label)
        
    if min_likes > 0:
        query += " AND c.likes >= ?"
        params.append(min_likes)
        
    query += " ORDER BY c.likes DESC, c.published_time DESC"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def get_video_leaderboard(channel_id):
    """Gets a list of all videos in the channel with comment counts and average sentiment."""
    conn = get_connection()
    query = """
    SELECT 
        v.video_id, 
        v.title, 
        v.published_time,
        COUNT(c.comment_id) as comment_count,
        AVG(c.sentiment_score) as avg_sentiment,
        SUM(CASE WHEN c.sentiment_label = 'Positive' THEN 1 ELSE 0 END) * 100.0 / COUNT(c.comment_id) as positive_pct,
        SUM(c.likes) as total_likes
    FROM videos v
    LEFT JOIN comments c ON v.video_id = c.video_id
    WHERE v.channel_id = ?
    GROUP BY v.video_id
    ORDER BY comment_count DESC
    """
    df = pd.read_sql_query(query, conn, params=[channel_id])
    conn.close()
    return df

def clear_channel(channel_id):
    """Deletes all data for a specific channel using cascade delete."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM channels WHERE channel_id = ?", (channel_id,))
    conn.commit()
    conn.close()
