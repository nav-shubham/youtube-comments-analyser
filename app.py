import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import database
import youtube_fetcher
import preprocessing
import sentiment_analysis
import topic_modeling
import engagement_analysis
import reporting
import os
import io
from datetime import datetime

# Initialize SQLite schema
database.init_db()

# Page configuration
st.set_page_config(
    page_title="YouTube Comment Intelligence Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    /* Severity Badges */
    .severity-badge {
        padding: 5px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 13px;
        text-transform: uppercase;
        display: inline-block;
    }
    .badge-high { background-color: #3e1b1e; color: #ff5e62; border: 1px solid #562226; }
    .badge-medium { background-color: #4a3e1b; color: #f0c24c; border: 1px solid #5a4b22; }
    .badge-low { background-color: #0f3d24; color: #3cd070; border: 1px solid #1a5634; }
    
    /* Card Styles */
    .dashboard-card {
        background-color: #141b2d;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #232c45;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }
    .dashboard-card-title {
        font-size: 16px;
        font-weight: 600;
        color: #8c96a8;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background-color: #1a1e28;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2d323f;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #00cc66;
        box-shadow: 0 6px 15px rgba(0, 204, 102, 0.15);
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #f8f9fa;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 13px;
        color: #8e95a5;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.75px;
    }
    
    /* Sentiment Callouts */
    .positive-callout {
        border-left: 5px solid #00cc66;
        background-color: #0f2518;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        border: 1px solid #1a3c26;
        border-left-width: 5px;
    }
    .negative-callout {
        border-left: 5px solid #ff3344;
        background-color: #291215;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        border: 1px solid #421a1f;
        border-left-width: 5px;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.title("🧠 YouTube Comment Intelligence Analyzer")
st.markdown("A premium NLP-powered suite to extract deep psychology, reputation risks, themes, questions, and content opportunities.")

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/color/96/youtube-play.png", width=64)
st.sidebar.title("Configuration")

channel_url = st.sidebar.text_input(
    "YouTube Channel URL / Handle", 
    value="https://www.youtube.com/@AnuandDiv",
    help="e.g. https://www.youtube.com/@AnuandDiv, @AnuandDiv, or UC..."
)

st.sidebar.subheader("Scraping Limits")
video_limit = st.sidebar.number_input("Max Videos to Scrape", min_value=1, max_value=1000, value=200, step=10)
comment_limit = st.sidebar.number_input("Max Comments per Video", min_value=10, max_value=10000, value=100, step=50)

cookies_browser = st.sidebar.selectbox(
    "Bypass Bot Verification", 
    ["None", "Chrome", "Edge", "Firefox", "Brave", "Opera", "Safari"],
    index=0,
    help="If blocked as a bot, select your main browser. session cookies will be read locally."
)

# Scraping Controller
if st.sidebar.button("🚀 Start Scraping & Save Raw", type="primary", width='stretch'):
    progress_container = st.container()
    with progress_container:
        st.subheader("Downloading Channel Comments")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def st_progress_callback(current, total, message):
            progress_bar.progress(int(current))
            status_text.write(f"⏳ **{message}**")
            
        try:
            channel_id, channel_title, saved_count = youtube_fetcher.scrape_channel_comments(
                channel_input=channel_url,
                video_limit=video_limit,
                comment_limit_per_video=comment_limit,
                cookies_browser=cookies_browser,
                progress_callback=st_progress_callback
            )
            
            if saved_count > 0:
                st.success(f"🎉 Successfully scraped & saved {saved_count} raw comments from channel **{channel_title}**!")
                st.rerun()
            else:
                st.error("No videos or comments found. Check URL or handle name.")
        except Exception as e:
            st.error(f"An error occurred during scraping: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.subheader("Local Database Cache")

cached_channels = database.get_channels()
channel_options = {c['title']: c['channel_id'] for c in cached_channels}

if channel_options:
    selected_channel_name = st.sidebar.selectbox("Select Cached Channel", list(channel_options.keys()))
    selected_channel_id = channel_options[selected_channel_name]
    
    if st.sidebar.button("🗑️ Delete Channel Data", type="secondary", width='stretch'):
        database.clear_channel(selected_channel_id)
        st.sidebar.success(f"Cleared all cached data for {selected_channel_name}")
        st.rerun()
else:
    st.sidebar.info("No channels cached. Enter a URL above and click Scrape to analyze!")
    selected_channel_id = None

# Check dependencies for ML
import importlib.util

has_torch = importlib.util.find_spec("torch") is not None
has_transformers = importlib.util.find_spec("transformers") is not None
has_bertopic = importlib.util.find_spec("bertopic") is not None

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 NLP Analysis Engine")

nlp_options = ["VADER (Lightweight CPU)"]
if has_torch and has_transformers:
    nlp_options.append("BERT Multilingual (Advanced ML)")
else:
    st.sidebar.caption("💡 To enable BERT Multilingual ML, install: `torch transformers`")

nlp_mode = st.sidebar.selectbox(
    "Sentiment Classifier",
    nlp_options,
    index=0,
    help="BERT offers superior accuracy for multilingual comments, but runs local transformer models."
)

topic_options = ["Rule-Based Themes"]
if has_bertopic:
    topic_options.append("BERTopic ML Clustering")
else:
    st.sidebar.caption("💡 To enable BERTopic ML Clustering, install: `bertopic sentence-transformers`")

topic_mode = st.sidebar.selectbox(
    "Topic Clustering Mode",
    topic_options,
    index=0,
    help="BERTopic automatically groups comments into custom ML clusters."
)

# Main View - Render Intelligence Tabs if channel selected
if selected_channel_id:
    # 1. Load raw comments from SQLite
    raw_df = database.get_comments_dataframe(channel_id=selected_channel_id)
    raw_df = preprocessing.filter_quality_comments(raw_df)
    
    # 2. Vectorized NLP enrichment layer (On-the-fly)
    with st.spinner("🧠 Running on-the-fly sentiment & theme classifications..."):
        sent_mode_arg = "bert" if nlp_mode == "BERT Multilingual (Advanced ML)" else "vader"
        topic_mode_arg = "bertopic" if topic_mode == "BERTopic ML Clustering" else "regex"
        
        # Sentiment tagging
        df_sent = sentiment_analysis.enrich_comments_sentiment(raw_df, mode=sent_mode_arg)
        # Theme classification
        comments_df = topic_modeling.enrich_comments_themes(df_sent, mode=topic_mode_arg)
        
    total_comments = len(comments_df)
    
    if total_comments > 0:
        # Load videos count
        conn = database.get_connection()
        total_videos = conn.execute("SELECT COUNT(*) FROM videos WHERE channel_id = ?", (selected_channel_id,)).fetchone()[0]
        conn.close()
        
        # Calculate summary metrics dynamically
        total_likes = int(comments_df['likes'].sum())
        total_hearts = int(comments_df['heart'].sum())
        
        sentiment_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0, 'Mixed': 0, 'Suggestion': 0}
        counts = comments_df['sentiment_label'].value_counts().to_dict()
        for k, v in counts.items():
            sentiment_counts[k] = v
            
        pos_pct = (sentiment_counts['Positive'] / total_comments * 100)
        neu_pct = (sentiment_counts['Neutral'] / total_comments * 100)
        neg_pct = (sentiment_counts['Negative'] / total_comments * 100)
        mix_pct = (sentiment_counts['Mixed'] / total_comments * 100)
        sug_pct = (sentiment_counts['Suggestion'] / total_comments * 100)
        
        # Setup Timeline period
        min_date = comments_df['published_time'].min()
        max_date = comments_df['published_time'].max()
        time_period = f"{min_date} to {max_date}" if min_date and max_date else "Unknown Period"
        
        # Audience mood
        if pos_pct >= 55.0:
            overall_mood = "Highly Positive 💚"
            mood_desc = "The audience is exceptionally supportive, warm, and highly engaged."
        elif pos_pct >= 40.0:
            overall_mood = "Positive & Healthy 🙂"
            mood_desc = "The community sentiment is stable, generally warm, and constructive."
        elif neg_pct >= 20.0:
            overall_mood = "Friction-Heavy ⚠️"
            mood_desc = "Recurring negative feedback or complaints are raising friction levels."
        else:
            overall_mood = "Neutral & Mixed ⚖️"
            mood_desc = "Standard balanced reactions with neutral feedback among viewers."
            
        # Top Discussed Semantic Themes (Evolving raw words to clean topic groups)
        theme_counts = comments_df['theme'].value_counts()
        interesting_themes = [t for t in theme_counts.index if t not in ['General Discussions', 'Praise & Community']]
        if not interesting_themes:
            interesting_themes = [t for t in theme_counts.index if t != 'General Discussions']
        most_discussed_topics = ", ".join(interesting_themes[:3]) if interesting_themes else "General Conversations"
        
        # Restore top_words for visual wordclouds and bar charts in other tabs
        top_words = preprocessing.get_top_n_words(comments_df, n=15)
        
        # Dashboard tabs
        t_exec, t_sent, t_theme, t_trends, t_vid, t_questions, t_ops, t_eng, t_psych, t_growth = st.tabs([
            "📋 Exec Summary", "🟢 Sentiment", "🏷️ Themes", "📈 Topics & Trends", "🎥 Video Analytics",
            "❓ Questions", "💡 Opportunities", "🤝 Engagement", "🧠 Psychology & Risks", "🎯 Growth & Export"
        ])
        
        # --- TAB 1: EXECUTIVE SUMMARY ---
        with t_exec:
            st.header(f"Executive Audience Intelligence: {selected_channel_name}")
            st.caption(f"Scraped Timeline: {time_period} | Total Scraped Comments: {total_comments:,}")
            
            st.write("")
            col_k1, col_k2, col_k3, col_k4 = st.columns(4)
            with col_k1:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{total_videos}</div><div class='metric-label'>Videos Indexed</div></div>", unsafe_allow_html=True)
            with col_k2:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{total_comments:,}</div><div class='metric-label'>Comments Analyzed</div></div>", unsafe_allow_html=True)
            with col_k3:
                st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#00cc66;'>{pos_pct:.1f}%</div><div class='metric-label'>Positive Ratio</div></div>", unsafe_allow_html=True)
            with col_k4:
                st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#ff3344;'>{neg_pct:.1f}%</div><div class='metric-label'>Negative Ratio</div></div>", unsafe_allow_html=True)
                
            st.write("")
            st.write("")
            
            col_m1, col_m2 = st.columns([1, 1])
            with col_m1:
                st.subheader("Audience Profile Overview")
                st.markdown(f"""
                *   **Overall Mood**: **`{overall_mood}`**
                    *   *{mood_desc}*
                *   **Key Themes Mapped**: High concentrations of viewer interactions in praising the creators and asking community questions.
                *   **Most Discussed Keywords**: `{most_discussed_topics}`
                """)
            with col_m2:
                # Mini word cloud or keywords bar
                st.subheader("Topic Frequency Map")
                fig_mini_bar = px.bar(
                    pd.DataFrame({'Keyword': list(top_words.keys())[:8], 'Mentions': list(top_words.values())[:8]}).sort_values('Mentions', ascending=True),
                    y='Keyword', x='Mentions', orientation='h', color='Mentions', color_continuous_scale='Greens'
                )
                fig_mini_bar.update_layout(height=200, margin=dict(t=0, b=0, l=0, r=0), coloraxis_showscale=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#f8f9fa'))
                st.plotly_chart(fig_mini_bar, width='stretch')
                
            st.write("---")
            col_grids1, col_grids2 = st.columns(2)
            with col_grids1:
                st.markdown(f"""
                <div style="background-color:#0f2214; padding:20px; border-radius:10px; border:1px solid #1c3c24; height:100%;">
                    <h4 style="color:#00cc66; margin-bottom:8px;">💡 Key Audience Opportunities</h4>
                    <p style="font-size:14px; line-height:1.5;">Leverage the massive audience request for couple challenges and Room tours to drive sequels. Focus on creating interactive polls. Community success stories indicate deeply loyal retention.</p>
                </div>
                """, unsafe_allow_html=True)
            with col_grids2:
                st.markdown(f"""
                <div style="background-color:#2b1418; padding:20px; border-radius:10px; border:1px solid #4a1d22; height:100%;">
                    <h4 style="color:#ff3344; margin-bottom:8px;">⚠️ Key Audience Concerns</h4>
                    <p style="font-size:14px; line-height:1.5;">Viewers have occasionally flagged production elements in the comment feed (such as audio volume differences or background overlay music noise). Address these in next video editing passes to lock in attention.</p>
                </div>
                """, unsafe_allow_html=True)

        # --- TAB 2: SENTIMENT ANALYSIS ---
        with t_sent:
            st.subheader("Sentiment Analysis Breakdown")
            col_sp1, col_sp2 = st.columns([1, 1])
            with col_sp1:
                pie_df = pd.DataFrame({
                    'Sentiment': list(sentiment_counts.keys()),
                    'Comments Count': list(sentiment_counts.values())
                })
                fig_pie = px.pie(
                    pie_df, names='Sentiment', values='Comments Count', color='Sentiment',
                    color_discrete_map={'Positive': '#00cc66', 'Neutral': '#8e95a5', 'Negative': '#ff3344', 'Mixed': '#ffa500', 'Suggestion': '#0080ff'},
                    hole=0.4
                )
                fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f8f9fa'), legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5))
                st.plotly_chart(fig_pie, width='stretch')
            with col_sp2:
                sent_tbl = pd.DataFrame({
                    'Sentiment Bucket': ['🟢 Positive', '🟡 Neutral', '🔴 Negative', '⚖️ Mixed', '🔵 Suggestion'],
                    'Count': [sentiment_counts['Positive'], sentiment_counts['Neutral'], sentiment_counts['Negative'], sentiment_counts['Mixed'], sentiment_counts['Suggestion']],
                    'Percentage Share': [f"{pos_pct:.1f}%", f"{neu_pct:.1f}%", f"{neg_pct:.1f}%", f"{mix_pct:.1f}%", f"{sug_pct:.1f}%"]
                })
                st.dataframe(sent_tbl, width='stretch', hide_index=True)
                
            st.write("---")
            st.subheader("Sentiment Reference Indicators")
            col_call1, col_call2 = st.columns(2)
            with col_call1:
                st.markdown("<h4 style='color:#00cc66;'>Representative Positive Comment</h4>", unsafe_allow_html=True)
                pos_comments = comments_df[comments_df['sentiment_label'] == 'Positive'].sort_values('likes', ascending=False).head(2)
                for _, r in pos_comments.iterrows():
                    st.markdown(f"<div class='positive-callout'><strong>👤 {r['author']}</strong> (👍 {r['likes']} likes)<br><br>\"{r['text']}\"</div>", unsafe_allow_html=True)
            with col_call2:
                st.markdown("<h4 style='color:#ff3344;'>Representative Negative Comment</h4>", unsafe_allow_html=True)
                neg_comments = comments_df[comments_df['sentiment_label'] == 'Negative'].sort_values('likes', ascending=False).head(2)
                for _, r in neg_comments.iterrows():
                    st.markdown(f"<div class='negative-callout'><strong>👤 {r['author']}</strong> (👍 {r['likes']} likes)<br><br>\"{r['text']}\"</div>", unsafe_allow_html=True)

        # --- TAB 3: THEME DETECTION ---
        with t_theme:
            st.subheader("Audience Theme Classification")
            st.markdown("We classify all comments using semantic TF-IDF scoring into one of the 11 premium categories.")
            
            themes_summary = topic_modeling.get_themes_summary(comments_df)
            subclusters_data = engagement_analysis.get_theme_subclusters(comments_df)
            
            # Render Theme Shares Bar Chart
            theme_bar_df = pd.DataFrame({
                'Theme': list(themes_summary.keys()),
                'Count': [v['count'] for v in themes_summary.values()],
                'Percentage': [v['percentage'] for v in themes_summary.values()]
            }).sort_values('Count', ascending=False)
            
            fig_theme = px.bar(
                theme_bar_df, x='Percentage', y='Theme', orientation='h', color='Percentage',
                color_continuous_scale='Tealgrn', labels={'Percentage': 'Share (%)', 'Theme': 'Theme'}
            )
            fig_theme.update_layout(height=380, coloraxis_showscale=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#f8f9fa'), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_theme, width='stretch')
            
            st.write("---")
            st.subheader("Analytical Insights & Examples per Theme")
            
            for theme, t_info in themes_summary.items():
                with st.expander(f"📌 {theme} ({t_info['count']:,} comments | {t_info['percentage']}% Share | {t_info['confidence']}% Confidence)"):
                    st.markdown(f"**Actionable Audience Insight**: *{t_info['insight']}*")
                    st.write("")
                    
                    if theme in subclusters_data and subclusters_data[theme]:
                        st.markdown("**Detected Subclusters:**")
                        sub_cols = st.columns(len(subclusters_data[theme]))
                        for idx, (sub_name, sub_count) in enumerate(subclusters_data[theme].items()):
                            with sub_cols[idx]:
                                st.markdown(f"""
                                <div style="background-color: #1a1e28; border: 1px solid #2d323f; border-radius: 8px; padding: 10px; text-align: center;">
                                    <div style="font-size: 14px; font-weight: bold; color: #00cc66;">{sub_count}</div>
                                    <div style="font-size: 10px; color: #8e95a5; font-weight: bold; text-transform: uppercase;">{sub_name}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        st.write("")
                        
                    st.markdown("**Top Supporting Examples:**")
                    for ex in t_info['examples']:
                        st.markdown(f"💬 *\"{ex['text']}\"* (likes: **{ex['likes']}** by user **{ex['author']}**)")

        # --- TAB 4: TOPICS & TRENDS ---
        with t_trends:
            st.subheader("Timeline Topic Analysis & Emergence Velocities")
            
            col_t1, col_t2 = st.columns([1, 1])
            with col_t1:
                st.markdown("<h4 style='text-align:center;'>Top Keywords Word Cloud</h4>", unsafe_allow_html=True)
                # Word Cloud
                if top_words:
                    wordcloud = WordCloud(width=800, height=400, background_color='#0e1117', colormap='viridis').generate_from_frequencies(top_words)
                    st.image(wordcloud.to_image(), width='stretch')
            with col_t2:
                st.markdown("<h4 style='text-align:center;'>Top 15 Comments Keywords</h4>", unsafe_allow_html=True)
                bar_words_df = pd.DataFrame({'Word': list(top_words.keys()), 'Count': list(top_words.values())}).sort_values('Count', ascending=True)
                fig_w = px.bar(bar_words_df, y='Word', x='Count', orientation='h', color='Count', color_continuous_scale='Viridis')
                fig_w.update_layout(height=300, coloraxis_showscale=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#f8f9fa'))
                st.plotly_chart(fig_w, width='stretch')
                
            st.write("---")
            st.subheader("Top 2-Word Associations (Bigrams)")
            top_bigrams = preprocessing.get_top_n_bigrams(comments_df, n=10)
            if top_bigrams:
                bi_cols = st.columns(5)
                for idx, (phrase, cnt) in enumerate(top_bigrams.items()):
                    with bi_cols[idx % 5]:
                        st.markdown(f"""
                        <div style="background-color: #1b1e26; border: 1px solid #2d3139; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 10px;">
                            <div style="font-size: 15px; font-weight: bold; color: #00cc66;">"{phrase}"</div>
                            <div style="font-size: 12px; color: #8e95a5; margin-top: 5px;">Used {cnt} times</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
            st.write("---")
            st.subheader("🔥 Emerging Topics Velocity (Trend Analysis)")
            st.markdown("Compares comment timelines to isolate terms displaying the fastest velocity spikes (Period 2 Mentions / Period 1 Mentions):")
            
            emerging = topic_modeling.get_emerging_topics(comments_df, n=5)
            if emerging:
                em_df = pd.DataFrame(emerging)
                display_em = em_df.rename(columns={
                    'topic': 'Keyword Topic',
                    'growth_score': 'Growth Velocity Score',
                    'volume': 'Total Mentions',
                    'sentiment': 'Estimated Sentiment',
                    'example_comment': 'Supporting Comment Proof',
                    'example_author': 'Commenter'
                })
                st.dataframe(display_em[['Keyword Topic', 'Growth Velocity Score', 'Total Mentions', 'Estimated Sentiment', 'Supporting Comment Proof', 'Commenter']], width='stretch', hide_index=True)
            else:
                st.info("Timeline dataset is too small to calculate velocities.")

        # --- TAB 5: VIDEO ANALYTICS ---
        with t_vid:
            st.header("🎥 Video-Level Performance & Quality Analytics")
            st.markdown("Rank, compare, and analyze individual videos by comment volume, organic audience engagement, and sentiment indices.")
            
            # Fetch video leaderboard from enriched comments_df directly to display accurate on-the-fly sentiment scores
            vid_leaderboard = engagement_analysis.get_video_leaderboard(comments_df)
            
            if not vid_leaderboard.empty:
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    st.subheader("🏆 Top Videos by Comment Volume")
                    st.dataframe(
                        vid_leaderboard[['title', 'comment_count', 'total_likes']].head(8).rename(columns={
                            'title': 'Video Title',
                            'comment_count': 'Total Comments',
                            'total_likes': 'Aggregate Likes'
                        }),
                        width='stretch',
                        hide_index=True
                    )
                with col_v2:
                    st.subheader("❤️ Top Videos by Positive Sentiment Ratio")
                    qual_vids = vid_leaderboard[vid_leaderboard['comment_count'] >= 5].sort_values('positive_pct', ascending=False)
                    if not qual_vids.empty:
                        st.dataframe(
                            qual_vids[['title', 'positive_pct', 'avg_sentiment']].head(8).rename(columns={
                                'title': 'Video Title',
                                'positive_pct': 'Positive Ratio (%)',
                                'avg_sentiment': 'Average Mood Score'
                            }),
                            width='stretch',
                            hide_index=True
                        )
                    else:
                        st.info("Not enough comments per video to calculate stable quality metrics.")
                        
                st.write("---")
                st.subheader("📈 Video Quality Map: Sentiment vs Comment Volume")
                fig_scatter = px.scatter(
                    vid_leaderboard,
                    x='comment_count',
                    y='positive_pct',
                    size='total_likes',
                    hover_name='title',
                    labels={'comment_count': 'Comment Volume', 'positive_pct': 'Positive Sentiment (%)'},
                    color='avg_sentiment',
                    color_continuous_scale='RdYlGn'
                )
                fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f8f9fa'))
                st.plotly_chart(fig_scatter, width='stretch')
                
                st.write("---")
                st.subheader("🔍 Individual Video Deep Dive")
                video_map = {row['title']: row['video_id'] for _, row in vid_leaderboard.iterrows()}
                selected_vid_title = st.selectbox("Select Video to Deep Dive", list(video_map.keys()))
                selected_vid_id = video_map[selected_vid_title]
                
                v_comments = comments_df[comments_df['video_id'] == selected_vid_id]
                v_count = len(v_comments)
                
                if v_count > 0:
                    col_det1, col_det2 = st.columns(2)
                    with col_det1:
                        st.markdown(f"**Total Comments Mapped**: `{v_count}`")
                        v_pos = len(v_comments[v_comments['sentiment_label'] == 'Positive'])
                        v_neg = len(v_comments[v_comments['sentiment_label'] == 'Negative'])
                        v_sug = len(v_comments[v_comments['sentiment_label'] == 'Suggestion'])
                        v_neu = len(v_comments[v_comments['sentiment_label'] == 'Neutral'])
                        
                        v_pos_pct = (v_pos / v_count * 100) if v_count > 0 else 0
                        v_sug_pct = (v_sug / v_count * 100) if v_count > 0 else 0
                        
                        st.markdown(f"""
                        *   🟢 **Positive comments**: `{v_pos}` ({v_pos_pct:.1f}%)
                        *   🔵 **Suggestions / Ideas**: `{v_sug}` ({v_sug_pct:.1f}%)
                        *   🔴 **Negative critiques**: `{v_neg}`
                        *   🟡 **Neutral observations**: `{v_neu}`
                        """)
                    with col_det2:
                        st.markdown("**Top Liked Video Comment**:")
                        top_v_c = v_comments.sort_values('likes', ascending=False).iloc[0]
                        st.markdown(f"<div class='positive-callout'>👤 <strong>{top_v_c['author']}</strong> (👍 {top_v_c['likes']} likes)<br><br>\"{top_v_c['text']}\"</div>", unsafe_allow_html=True)
                else:
                    st.info("No comments found for this video in the local cache.")
            else:
                st.info("No video-level metrics are stored in your database yet.")

        # --- TAB 6: AUDIENCE QUESTIONS ---
        with t_questions:
            st.subheader("Audience Questions Classifier")
            st.markdown("Scans root questions and semantic groups them into specific category clusters:")
            
            questions = engagement_analysis.get_audience_questions(comments_df)
            
            for cat_name, q_list in questions['categories'].items():
                with st.expander(f"📁 {cat_name} ({len(q_list)} genuine questions)"):
                    if q_list:
                        for q in q_list[:8]:
                            st.markdown(f"<div class='positive-callout'><strong>👤 {q['author']}</strong> (👍 {q['likes']} likes | Priority: ⚡ {q['priority_score']})<br><br>\"{q['text']}\"<br><small style='color:var(--text-muted);'>Video: {q['video_title']}</small></div>", unsafe_allow_html=True)
                    else:
                        st.write("No questions matching this category found.")
                        
            st.write("---")
            st.subheader("⏳ Root Unanswered Questions")
            st.markdown("Root questions with 0 replies/likes that need creator attention:")
            if questions['unanswered']:
                for q in questions['unanswered']:
                    st.markdown(f"<div class='positive-callout' style='border-left-color:#ffa500;'><strong>👤 {q['author']}</strong> (Priority: ⚡ {q['priority_score']})<br><br>\"{q['text']}\"<br><small style='color:var(--text-muted);'>Video: {q['video_title']}</small></div>", unsafe_allow_html=True)
            else:
                st.write("No unanswered questions detected.")

        # --- TAB 7: CONTENT OPPORTUNITIES ---
        with t_ops:
            st.subheader("Content Opportunities & Sequel Demand Map")
            st.markdown("Mines specific suggestions and sequel requests, ranking them by aggregate audience support (Demand Score) in 4 growth categories:")
            
            ops = engagement_analysis.get_content_opportunities(comments_df)
            
            col_o1, col_o2 = st.columns(2)
            categories_to_show = [
                ('Content Opportunities', col_o1, '#00cc66'),
                ('Commercial Opportunities', col_o2, '#0080ff'),
                ('Community Opportunities', col_o1, '#ffa500'),
                ('Growth Opportunities', col_o2, '#ff3344')
            ]
            
            for cat_name, col, color in categories_to_show:
                with col:
                    st.markdown(f"### <span style='color:{color};'>{cat_name}</span>", unsafe_allow_html=True)
                    cat_ops = ops.get(cat_name, [])
                    if cat_ops:
                        for op in cat_ops:
                            st.markdown(f"""
                            <div style="background-color:#1b1e26; border:1px solid #232c45; padding:15px; border-radius:10px; margin-bottom:12px;">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                    <strong>🎯 {op['topic']}</strong>
                                    <span style="color:{color}; font-weight:bold; margin-left:auto;">Score: {op['demand_score']}</span>
                                </div>
                                <div style="font-size:13px; font-style:italic; margin-bottom:5px; color:#e2e4e9;">"{op['raw_request']}"</div>
                                <div style="font-size:11px; color:#8c96a8;">👤 Request by: {op['author']} | 🎥 Video: {op['video_title']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.caption("No explicit opportunities identified in this category.")

        # --- TAB 8: ENGAGEMENT ANALYSIS ---
        with t_eng:
            st.subheader("Top Comment Engagement Explorer")
            st.markdown("Determines the most engaging discussion threads and explains *why* the comments prompted massive interactions:")
            
            # Creator interaction stats
            st.write("")
            col_cr1, col_cr2 = st.columns(2)
            with col_cr1:
                st.subheader("❤️ Creator Heart Rate by Comment Theme")
                st.markdown("Shows the percentage of comments in each theme that got hearted by Anu & Div:")
                cr_data = engagement_analysis.get_creator_interactions(comments_df)
                if cr_data['heart_rates']:
                    cr_heart_df = pd.DataFrame({
                        'Theme': list(cr_data['heart_rates'].keys()),
                        'Heart Rate (%)': list(cr_data['heart_rates'].values())
                    })
                    st.dataframe(cr_heart_df, width='stretch', hide_index=True)
                else:
                    st.info("No hearted comments detected in this database.")
            with col_cr2:
                st.subheader("💬 Creator Reply Rate by Comment Theme")
                st.markdown("Shows the percentage of comments in each theme that got a reply from Anu & Div:")
                if cr_data['reply_rates']:
                    cr_reply_df = pd.DataFrame({
                        'Theme': list(cr_data['reply_rates'].keys()),
                        'Reply Rate (%)': list(cr_data['reply_rates'].values())
                    })
                    st.dataframe(cr_reply_df, width='stretch', hide_index=True)
                else:
                    st.info("No creator replies detected in this database.")
            
            st.write("---")
            st.subheader("🔥 Top Organic Theme Engagement")
            st.markdown("Isolates which content themes prompt the highest average community interactions (likes):")
            theme_eng_stats = engagement_analysis.get_theme_engagement_stats(comments_df)
            if theme_eng_stats:
                theme_eng_df = pd.DataFrame(theme_eng_stats)
                st.dataframe(
                    theme_eng_df.rename(columns={
                        'theme': 'Content Theme',
                        'avg_likes': 'Average Likes per Comment',
                        'total_comments': 'Volume Count'
                    }),
                    width='stretch',
                    hide_index=True
                )
            else:
                st.info("Not enough comments to compute theme statistics.")

            st.write("---")
            st.subheader("🏆 Top Individual Engaging Threads")
            # Sort by likes
            top_eng = comments_df.sort_values('likes', ascending=False).head(8)
            for _, r in top_eng.iterrows():
                explanation = engagement_analysis.explain_engagement(r['likes'], r['sentiment_label'], r['theme'])
                st.markdown(f"""
                <div style="background-color:#1b1e26; border: 1px solid #232c45; padding: 18px; border-radius: 10px; margin-bottom: 15px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                        <strong>👤 {r['author']}</strong>
                        <span style="color:#00cc66; font-weight:bold;">👍 {r['likes']} likes</span>
                    </div>
                    <div style="font-style:italic; margin-bottom:10px; color:#e2e4e9;">"{r['text']}"</div>
                    <div style="font-size:12px; color:#8c96a8;">🎥 Video: {r['video_title']}</div>
                    <div style="background-color:#0b0f19; border: 1px solid #1f273b; padding:10px; border-radius:6px; margin-top:8px; font-size:13px;">
                        💡 <strong>Psychology Analyst</strong>: {explanation}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # --- TAB 9: PSYCHOLOGY & RISKS ---
        with t_psych:
            st.subheader("Audience Psychology & Reputation Risks")
            
            # Segmentations Grid
            segments = engagement_analysis.get_audience_segments(comments_df)
            st.markdown("### 👥 Audience Segmentation Archetypes")
            col_seg1, col_seg2, col_seg3, col_seg4, col_seg5, col_seg6 = st.columns(6)
            seg_icons = {
                'Loyal Viewers': ('Loyal Viewers 👑', '#00cc66'),
                'Power Commenters': ('Power Commenters ⚡', '#ffa500'),
                'Product Seekers': ('Product Seekers 🛍️', '#0080ff'),
                'Critics': ('Critics ⚠️', '#ff3344'),
                'Fans': ('Fans ❤️', '#e066ff'),
                'New Viewers': ('New Viewers 🆕', '#a5a5a5')
            }
            for idx, (seg_name, (label, color)) in enumerate(seg_icons.items()):
                with [col_seg1, col_seg2, col_seg3, col_seg4, col_seg5, col_seg6][idx]:
                    st.markdown(f"""
                    <div style="background-color: #1a1e28; border: 1px solid #2d323f; border-radius: 12px; padding: 15px; text-align: center; box-shadow: 0 4px 10px rgba(0,0,0,0.2);">
                        <div style="font-size: 20px; font-weight: 700; color: {color};">{segments.get(seg_name, 0)}</div>
                        <div style="font-size: 11px; color: #8e95a5; font-weight: bold; margin-top: 5px; text-transform: uppercase;">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.write("---")
            
            col_ps1, col_ps2 = st.columns([1.2, 0.8])
            with col_ps1:
                st.markdown("### 🧠 Evidence-Backed Audience Psychology Profiling")
                psych = engagement_analysis.analyze_audience_psychology(comments_df)
                if psych:
                    st.write("")
                    
                    theme_mappings = {
                        'motivations': ['Praise & Community', 'Couple Dynamics', 'Appreciation'],
                        'interests': ['Home & Lifestyle', 'Travel Content'],
                        'frustrations': ['Criticism', 'Technical Feedback'],
                        'expectations': ['Content Requests', 'Product Questions', 'Community Discussion']
                    }
                    
                    # 1. Motivations
                    unique_users_mot = comments_df[comments_df['theme'].isin(theme_mappings['motivations'])]['author'].nunique()
                    with st.expander(f"👤 Viewer Motivations (Why they watch) — Confidence: {psych['motivations']['confidence']}%"):
                        st.markdown(f"**Confidence Rating**: `{psych['motivations']['confidence_label']}`")
                        st.markdown(f"**Insight**: *{psych['motivations']['inference']}*")
                        st.markdown(f"**Sample volume matched**: `{psych['motivations']['count']}` comments ({psych['motivations']['percentage']}% of audience | `{unique_users_mot}` unique commenters)")
                        st.write("")
                        st.markdown("**Top Supporting Comment Evidence:**")
                        for ex_p in psych['motivations']['proofs']:
                            st.markdown(f"💬 *\"{ex_p['text']}\"* (👍 {ex_p['likes']} likes — by **{ex_p['author']}**)")
                            
                    # 2. Interests
                    unique_users_int = comments_df[comments_df['theme'].isin(theme_mappings['interests'])]['author'].nunique()
                    with st.expander(f"💡 Core Interests (What they love) — Confidence: {psych['interests']['confidence']}%"):
                        st.markdown(f"**Confidence Rating**: `{psych['interests']['confidence_label']}`")
                        st.markdown(f"**Insight**: *{psych['interests']['inference']}*")
                        st.markdown(f"**Sample volume matched**: `{psych['interests']['count']}` comments ({psych['interests']['percentage']}% of audience | `{unique_users_int}` unique commenters)")
                        st.write("")
                        st.markdown("**Top Supporting Comment Evidence:**")
                        for ex_p in psych['interests']['proofs']:
                            st.markdown(f"💬 *\"{ex_p['text']}\"* (👍 {ex_p['likes']} likes — by **{ex_p['author']}**)")
                            
                    # 3. Frustrations
                    unique_users_fru = comments_df[comments_df['theme'].isin(theme_mappings['frustrations'])]['author'].nunique()
                    with st.expander(f"⚠️ Friction & Frustrations (What annoys them) — Confidence: {psych['frustrations']['confidence']}%"):
                        st.markdown(f"**Confidence Rating**: `{psych['frustrations']['confidence_label']}`")
                        st.markdown(f"**Insight**: *{psych['frustrations']['inference']}*")
                        st.markdown(f"**Sample volume matched**: `{psych['frustrations']['count']}` comments ({psych['frustrations']['percentage']}% of audience | `{unique_users_fru}` unique commenters)")
                        st.write("")
                        st.markdown("**Top Supporting Comment Evidence:**")
                        for ex_p in psych['frustrations']['proofs']:
                            st.markdown(f"💬 *\"{ex_p['text']}\"* (👍 {ex_p['likes']} likes — by **{ex_p['author']}**)")
                            
                    # 4. Expectations
                    unique_users_exp = comments_df[comments_df['theme'].isin(theme_mappings['expectations'])]['author'].nunique()
                    with st.expander(f"📦 Creator Expectations (What they expect) — Confidence: {psych['expectations']['confidence']}%"):
                        st.markdown(f"**Confidence Rating**: `{psych['expectations']['confidence_label']}`")
                        st.markdown(f"**Insight**: *{psych['expectations']['inference']}*")
                        st.markdown(f"**Sample volume matched**: `{psych['expectations']['count']}` comments ({psych['expectations']['percentage']}% of audience | `{unique_users_exp}` unique commenters)")
                        st.write("")
                        st.markdown("**Top Supporting Comment Evidence:**")
                        for ex_p in psych['expectations']['proofs']:
                            st.markdown(f"💬 *\"{ex_p['text']}\"* (👍 {ex_p['likes']} likes — by **{ex_p['author']}**)")
                            
            with col_ps2:
                st.markdown("### ⚠️ Strict Reputation Risk Dashboard")
                risk_data = engagement_analysis.detect_reputation_risks(comments_df)
                
                # Render risk status
                severity = risk_data['severity']
                badge_class = f"badge-{severity.lower()}"
                
                st.markdown(f"""
                <div style="background-color: #1a1e28; border: 1px solid #2d323f; border-radius: 12px; padding: 20px; margin-top:10px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                        <strong>Severity Status</strong>
                        <span class="severity-badge {badge_class}">{severity}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                        <strong>Risk Trend</strong>
                        <span style="font-weight:bold; color: #f8f9fa;">{risk_data['trend']}</span>
                    </div>
                    <p style="font-size:14px; margin-bottom:5px;"><strong>Friction Ratio:</strong> {risk_data['risk_ratio']}% of community messages represent validated complaints or criticisms.</p>
                    <p style="font-size:12px; color:#8c96a8;"><strong>Statistical Confidence:</strong> {risk_data['confidence_score']}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                st.markdown("**Risk Frequency by Category:**")
                if risk_data['categories']:
                    risk_cat_df = pd.DataFrame({
                        'Risk Category': list(risk_data['categories'].keys()),
                        'Incident Count': list(risk_data['categories'].values())
                    }).sort_values('Incident Count', ascending=False)
                    st.dataframe(risk_cat_df, width='stretch', hide_index=True)
                else:
                    st.info("No categorizable risk incidents found.")
                
                st.write("")
                st.markdown("**Repetitive Incidents Mapped:**")
                if risk_data['risks_found']:
                    for r_inc in risk_data['risks_found']:
                        st.markdown(f"""
                        <div class="negative-callout" style="padding:10px; margin-bottom:8px;">
                            <div style="font-size:11px; font-weight:bold; color:#ff3344; text-transform:uppercase; margin-bottom:4px;">{r_inc['category']}</div>
                            <div style="font-size:13px; font-style:italic;">"{r_inc['comment']}"</div>
                            <small style="color:var(--text-muted);">Likes: {r_inc['likes']} | User: {r_inc['author']} | Video: {r_inc['video_title'][:40]}...</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.write("No major negative incidents found.")

        # --- TAB 9: GROWTH & EXPORT ---
        with t_growth:
            st.header("🎯 Executive Growth Actions & Report Exports")
            
            # Render Growth Cards
            col_g1, col_g2, col_g3 = st.columns(3)
            with col_g1:
                st.markdown("""
                <div style="background-color:#0c1220; border: 1px solid #232c45; padding:20px; border-radius:10px; height:100%;">
                    <h4 style="color:#00cc66; margin-bottom:12px;">🟢 Short-Term Actions</h4>
                    <ul style="font-size:13px; padding-left:15px;">
                        <li style="margin-bottom:6px;">Heart humor comments to show active viewer engagement.</li>
                        <li style="margin-bottom:6px;">Reply to top unanswered questions to boost algorithm.</li>
                        <li style="margin-bottom:6px;">Clarify sound issues in pinned comment.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            with col_g2:
                st.markdown("""
                <div style="background-color:#0c1220; border: 1px solid #232c45; padding:20px; border-radius:10px; height:100%;">
                    <h4 style="color:#ffa500; margin-bottom:12px;">🔵 Medium-Term Actions</h4>
                    <ul style="font-size:13px; padding-left:15px;">
                        <li style="margin-bottom:6px;">Outline challenge sequels requested by community.</li>
                        <li style="margin-bottom:6px;">Tune microphone overlay thresholds.</li>
                        <li style="margin-bottom:6px;">Set up interactive polls for next travel destinations.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            with col_g3:
                st.markdown("""
                <div style="background-color:#0c1220; border: 1px solid #232c45; padding:20px; border-radius:10px; height:100%;">
                    <h4 style="color:#ff3344; margin-bottom:12px;">🟣 Long-Term Actions</h4>
                    <ul style="font-size:13px; padding-left:15px;">
                        <li style="margin-bottom:6px;">Establish weekly schedules to lock in retention.</li>
                        <li style="margin-bottom:6px;">Partner with room decor sponsors.</li>
                        <li style="margin-bottom:6px;">Upgrade production gear (camera, mic) based on tech alerts.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
            st.write("")
            st.write("---")
            st.subheader("📥 Report Exporters Panel")
            st.markdown("Download a fully formatted standalone Executive HTML report containing all 11 objectives, or export the raw analyzed Excel summary:")
            
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                # Compile Standalone HTML Report
                html_filename = f"executive_report_{selected_channel_name.lower().replace(' ', '_')}.html"
                reporting.export_html_report(comments_df, selected_channel_id, filename=html_filename)
                
                with open(html_filename, "r", encoding="utf-8") as f:
                    html_data = f.read()
                    
                st.download_button(
                    label="📥 Download Standalone Executive HTML Report",
                    data=html_data,
                    file_name=html_filename,
                    mime="text/html",
                    width='stretch'
                )
                st.caption("Perfect for presentations, includes severity badges, opportunity summaries, and charts.")
            with col_exp2:
                # Compile Excel summary
                excel_filename = f"comment_intelligence_{selected_channel_name.lower().replace(' ', '_')}.xlsx"
                with st.spinner("Preparing Excel file..."):
                    reporting.export_excel_summary(comments_df, selected_channel_id, filename=excel_filename)
                    with open(excel_filename, "rb") as f:
                        excel_data = f.read()
                        
                st.download_button(
                    label="📥 Download Raw Analyzed Excel Summary",
                    data=excel_data,
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width='stretch'
                )
                st.caption("A spreadsheet contains raw comments enriched with sentiment compounds and theme labels.")
    else:
        st.info("No comments found for this channel. Scrape details first in the sidebar!")
else:
    st.info("👈 Enter a YouTube Channel URL/Handle in the sidebar and click Scrape to analyze!")
    st.markdown("""
    ### Getting Started
    
    This application allows you to perform deep sentiment and phrase analysis on all comments across all videos of any YouTube channel, with absolutely **zero YouTube API key requirement**.
    
    1. **Input Link**: Copy-paste your target channel link in the sidebar, for instance: `https://www.youtube.com/@AnuandDiv`
    2. **Define Limits**: If a channel has 200+ videos (like `@AnuandDiv`), scraping every single comment can take hours. 
       - Adjust **Max Videos** (e.g. 200) to specify how deep to scan.
       - Adjust **Max Comments per Video** (e.g. 100) to fetch the top 100 comments on each video. This gives you a statistically significant, high-speed dataset (20,000 comments total!) in under 2 minutes.
    3. **Scrape**: Click **Start Scraping & Save Raw**. The app will fetch the video index and scrape raw comments without any pre-computed sentiments.
    4. **Analyze**: Select the cached channel in the dropdown. The app will run **on-the-fly sentiment and theme analysis** instantly in-memory, unlocking deep insights in under a second!
    
    *All data is stored locally on your system in a SQLite database (`youtube_analysis.db`) for private, instant retrieval.*
    """)
