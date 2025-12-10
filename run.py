#!/usr/bin/env python3
"""
Entry point for the Automated Legal News Blog Generator.

Usage:
    python run.py

Environment variables required:
    GEMINI_API_KEY - Google Gemini API key
    SANITY_PROJECT_ID - Sanity.io project ID
    SANITY_TOKEN - Sanity.io API token
    SANITY_DATASET - Sanity dataset (default: production)

Optional:
    ENABLE_IMAGE_GENERATION - Set to 'false' to disable image generation
    DEFAULT_AUTHOR - Author name for posts (default: Case Value Expert)
"""

import time
from datetime import datetime, timezone

from auto_post import (
    scrape_all_sources,
    select_best_articles,
    generate_article,
    generate_article_from_title,
    get_existing_posts,
    post_to_sanity,
    load_title_list,
    save_title_list,
    load_used_topics,
    add_used_topic,
)
from auto_post.config import GEMINI_API_KEY, SANITY_PROJECT_ID, SANITY_TOKEN


def main():
    """Main execution flow."""
    print("=" * 60)
    print("  Legal News Blog Post Generator")
    print("  Personal Injury | Employment Law | Mass Torts")
    print(f"  Started at: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # Validate configuration
    missing_config = []
    if not GEMINI_API_KEY:
        missing_config.append("GEMINI_API_KEY")
    if not SANITY_PROJECT_ID:
        missing_config.append("SANITY_PROJECT_ID")
    if not SANITY_TOKEN:
        missing_config.append("SANITY_TOKEN")

    if missing_config:
        print(f"\nWarning: Missing environment variables: {', '.join(missing_config)}")
        print("Some features may not work correctly.")

    # Step 1: Scrape All News Sources
    print("\n--- Step 1: Scraping News Sources ---")
    all_news = scrape_all_sources()

    if not all_news:
        print("No articles scraped from any source. Exiting.")
        return

    # Step 2: Load used topics and select best articles
    print("\n--- Step 2: Selecting Articles (filtering for today's news and avoiding duplicates) ---")
    used_topics = load_used_topics()
    print(f"Loaded {len(used_topics)} previously covered topics to avoid duplicates")

    selected_articles = select_best_articles(all_news, num_articles=2, used_topics=used_topics)

    if not selected_articles:
        print("No suitable NEW articles from today found that match our criteria. Exiting.")
        print("The script will try again tomorrow when fresh news is available.")
        return

    print(f"\nSelected {len(selected_articles)} articles for blog generation:")
    for i, article in enumerate(selected_articles, 1):
        print(f"  {i}. {article['title'][:60]}...")
        print(f"     Source: {article['source']} | Category: {article['category']}")

    # Step 3: Get Internal Links
    print("\n--- Step 3: Fetching Internal Links ---")
    link_database = get_existing_posts()

    # Step 4 & 5: Generate and Post Each Article
    success_count = 0
    fail_count = 0

    for i, selected_article in enumerate(selected_articles, 1):
        print(f"\n{'='*60}")
        print(f"  Processing Article {i} of {len(selected_articles)}")
        print(f"{'='*60}")

        # Step 4: Generate Article
        print(f"\n--- Step 4.{i}: Generating Article ---")
        generated_article = generate_article(selected_article, all_news, link_database)

        if not generated_article:
            print(f"Article {i} generation failed. Skipping.")
            fail_count += 1
            continue

        # Display generated content summary
        print(f"\n  Generated Content:")
        print(f"  - Title: {generated_article.get('title', 'N/A')}")
        print(f"  - Slug: {generated_article.get('slug', 'N/A')}")
        print(f"  - Meta Title ({len(generated_article.get('meta_title', ''))} chars): {generated_article.get('meta_title', 'N/A')}")
        print(f"  - Excerpt ({len(generated_article.get('excerpt', ''))} chars): {generated_article.get('excerpt', 'N/A')[:80]}...")
        print(f"  - Categories: {', '.join(generated_article.get('categories', []))}")
        print(f"  - Keywords: {', '.join(generated_article.get('keywords', []))}")
        print(f"  - Alt Text: {generated_article.get('alt_text', 'N/A')[:60]}...")

        # Step 5: Post to Sanity
        print(f"\n--- Step 5.{i}: Posting to Sanity ---")
        success = post_to_sanity(generated_article)

        if success:
            success_count += 1
            # Track this topic to avoid duplicates in future runs
            topic_summary = selected_article.get('topic_summary', selected_article.get('title', '')[:100])
            if topic_summary:
                add_used_topic(topic_summary)
                print(f"  Added topic to tracking: {topic_summary[:50]}...")
        else:
            fail_count += 1

        # Small delay between posts
        if i < len(selected_articles):
            time.sleep(2)

    # Step 6: Generate article from pre-defined title list
    print(f"\n{'='*60}")
    print("  Processing Pre-defined Title Article")
    print(f"{'='*60}")

    titles = load_title_list()
    if titles:
        current_title = titles[0]
        print(f"\n--- Step 6: Generating Article from Title ---")
        print(f"Title: {current_title[:60]}...")

        title_article = generate_article_from_title(current_title, link_database)

        if title_article:
            # Display generated content summary
            print(f"\n  Generated Content:")
            print(f"  - Title: {title_article.get('title', 'N/A')}")
            print(f"  - Slug: {title_article.get('slug', 'N/A')}")
            print(f"  - Meta Title ({len(title_article.get('meta_title', ''))} chars): {title_article.get('meta_title', 'N/A')}")
            print(f"  - Excerpt ({len(title_article.get('excerpt', ''))} chars): {title_article.get('excerpt', 'N/A')[:80]}...")
            print(f"  - Categories: {', '.join(title_article.get('categories', []))}")
            print(f"  - Keywords: {', '.join(title_article.get('keywords', []))}")

            print(f"\n--- Posting Title Article to Sanity ---")
            title_success = post_to_sanity(title_article)

            if title_success:
                success_count += 1
                # Remove the used title and save
                titles.pop(0)
                save_title_list(titles)
            else:
                fail_count += 1
        else:
            print("Title article generation failed.")
            fail_count += 1
    else:
        print("\nNo pre-defined titles remaining in titles.json")

    print("\n" + "=" * 60)
    print(f"  COMPLETE: {success_count} blog post(s) published successfully!")
    if fail_count > 0:
        print(f"  FAILED: {fail_count} blog post(s) could not be published.")
    print("=" * 60)


if __name__ == "__main__":
    main()
