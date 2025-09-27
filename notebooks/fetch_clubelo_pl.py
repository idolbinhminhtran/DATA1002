"""
Fetch ClubElo histories for all teams present in a Premier League CSV and save a single CSV.

This DOES NOT join to your matches table. It only creates an Elo history table you can join later.

Example:
  python fetch_clubelo_pl.py \
    --in_csv PLdata-10-years.csv \
    --date_col Date \
    --home_col HomeTeam \
    --away_col AwayTeam \
    --out_csv clubelo_premierleague_history.csv

You can also pass teams directly (bypassing the input CSV) with --teams "Arsenal,Chelsea,Liverpool".
Requires: requests, pandas
"""
import argparse
import io
from functools import lru_cache

import pandas as pd
import requests

CLUBELO_BASE = "http://api.clubelo.com"

TEAM_SLUGS = {
    "Manchester City": "ManCity",
    "Man City": "ManCity",
    "Manchester Utd": "ManUnited",
    "Manchester United": "ManUnited",
    "Man United": "ManUnited",
    "Liverpool": "Liverpool",
    "Chelsea": "Chelsea",
    "Arsenal": "Arsenal",
    "Tottenham": "Tottenham",
    "Spurs": "Tottenham",
    "Newcastle": "Newcastle",
    "Newcastle United": "Newcastle",
    "Aston Villa": "AstonVilla",
    "West Ham": "WestHam",
    "West Ham United": "WestHam",
    "Leicester": "Leicester",
    "Leicester City": "Leicester",
    "Everton": "Everton",
    "Wolves": "Wolves",
    "Wolverhampton": "Wolves",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Southampton": "Southampton",
    "Crystal Palace": "CrystalPalace",
    "Bournemouth": "Bournemouth",
    "Fulham": "Fulham",
    "Brentford": "Brentford",
    "Burnley": "Burnley",
    "Leeds": "Leeds",
    "Leeds United": "Leeds",
    "Nottingham Forest": "Forest",
    "Nott'm Forest": "Forest",
    "Nottm Forest": "Forest",
    "Notts Forest": "Forest",
    "Forest": "Forest",
    "Sheffield United": "SheffieldUnited",
    "Sheffield Utd": "SheffieldUnited",
    "Sheffield Wednesday": "SheffieldWed",
    "Norwich": "Norwich",
    "Norwich City": "Norwich",
    "Watford": "Watford",
    "West Brom": "WestBrom",
    "West Bromwich": "WestBrom",
    "West Bromwich Albion": "WestBrom",
    "Cardiff": "Cardiff",
    "Swansea": "Swansea",
    "Huddersfield": "Huddersfield",
    "QPR": "QPR",
    "Queens Park Rangers": "QPR",
    "Hull": "Hull",
    "Hull City": "Hull",
    "Middlesbrough": "Middlesbrough",
    "Birmingham": "Birmingham",
    "Portsmouth": "Portsmouth",
    "Stoke": "Stoke",
    "Stoke City": "Stoke",
    "Sunderland": "Sunderland",
    "Blackburn": "Blackburn",
    "Wigan": "Wigan",
    "Reading": "Reading",
    "Ipswich": "Ipswich",
    "Derby": "Derby",
    "Bolton": "Bolton",
    "Charlton": "Charlton",
    "Coventry": "Coventry",
    "Fulham FC": "Fulham",
}

def normalise_team_name(name: str) -> str:
    if pd.isna(name):
        return name
    s = str(name).strip()
    return TEAM_SLUGS.get(s, s.replace(" ", "").replace("&", "").replace(".", ""))

@lru_cache(maxsize=None)
def fetch_team_csv(team_slug: str) -> pd.DataFrame:
    url = f"{CLUBELO_BASE}/{team_slug}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    # Normalize columns
    df.columns = [c.strip().lower() for c in df.columns]
    # Keep the essentials
    keep = [c for c in ["club", "from", "to", "elo", "rank", "country", "level"] if c in df.columns]
    df = df[keep].copy()
    df["team_slug"] = team_slug
    return df

def get_team_list(args) -> list[str]:
    if args.teams:
        return [t.strip() for t in args.teams.split(",") if t.strip()]
    # otherwise pull from CSV
    df = pd.read_csv(args.in_csv)
    teams = pd.concat([df[args.home_col], df[args.away_col]], axis=0).dropna().unique().tolist()
    return teams

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", help="Input PL matches CSV (used to enumerate teams)")
    ap.add_argument("--date_col", default="Date", help="Ignored here; kept for parity with join script")
    ap.add_argument("--home_col", default="HomeTeam", help="Home team column in input CSV")
    ap.add_argument("--away_col", default="AwayTeam", help="Away team column in input CSV")
    ap.add_argument("--out_csv", required=True, help="Output CSV path for consolidated Elo histories")
    ap.add_argument("--teams", help='Optional: comma-separated team names to fetch (bypass input CSV)')
    args = ap.parse_args()

    if not args.teams and not args.in_csv:
        raise SystemExit("Provide either --teams or --in_csv")

    teams = get_team_list(args)
    frames = []
    failed = []

    for name in sorted(set(teams)):
        slug = normalise_team_name(name)
        try:
            df = fetch_team_csv(slug)
            df["team_name"] = name  # original name from your set
            frames.append(df)
            print(f"[OK] {name} -> {slug} ({len(df)} rows)")
        except Exception as e:
            print(f"[WARN] {name} -> {slug} failed: {e}")
            failed.append((name, slug, str(e)))

    if not frames:
        raise SystemExit("No data fetched. Check connectivity or mappings.")

    out = pd.concat(frames, ignore_index=True)

    # Friendly column order
    cols = [c for c in ["team_name", "team_slug", "club", "country", "level", "from", "to", "elo", "rank"] if c in out.columns]
    out = out[cols]

    out.to_csv(args.out_csv, index=False)
    print(f"\n✅ Saved Elo histories: {args.out_csv}  (rows={len(out)})")

    if failed:
        print("\nTeams that failed to fetch (fix mapping or retry):")
        for name, slug, msg in failed:
            print(f"  - {name} -> {slug}: {msg}")

if __name__ == "__main__":
    main()