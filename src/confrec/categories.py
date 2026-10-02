"""Single source of truth: Lumen domain key -> Amazon-Reviews-2023 category file stem.

Imported by scripts/sigir/slim_amazon2023.py (download) and src/confrec/build_rated_panels.py (panels),
so a domain that can be built can always be downloaded.
"""
CATEGORY = {
    "sports": "Sports_and_Outdoors",
    "toys": "Toys_and_Games",
    "home": "Home_and_Kitchen",
    "tools": "Tools_and_Home_Improvement",
    "games": "Video_Games",
    "beauty": "All_Beauty",            # too sparse for rated panels (37 eligible users) — kept for Lumen
    "beauty_pc": "Beauty_and_Personal_Care",
    "books": "Books",
    "electronics": "Electronics",
    "movies": "Movies_and_TV",
    "cds": "CDs_and_Vinyl",
}
