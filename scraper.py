import praw
import os
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

# Load credentials
load_dotenv()

# Authenticate with Reddit API
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# Choose subreddit and number of posts
subreddit = reddit.subreddit("depression")
limit = 200  # You can increase to 500 or 1000 later

# List to store post data
posts = []

# Fetch posts
for post in subreddit.hot(limit=limit):
    posts.append({
        "title": post.title,
        "selftext": post.selftext,
        "score": post.score,
        "created_utc": datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
        "num_comments": post.num_comments,
        "url": post.url
    })

# Save to CSV
df = pd.DataFrame(posts)
df.to_csv("reddit_depression.csv", index=False, encoding='utf-8')

print(f"✅ Scraped and saved {len(df)} posts to reddit_depression.csv")
