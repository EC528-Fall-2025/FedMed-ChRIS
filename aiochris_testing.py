#!/usr/bin/env python3
# create_feed_and_predict.py — make a new feed, then run chrNIST off it (aiochris)

import asyncio
import sys
from aiochris import ChrisClient  # requires Python >= 3.11

SERVER = "http://localhost:8000/api/v1"
USER   = "chris"
PASS   = "chris1234"

# Plugins by name (IDs vary per CUBE)
PL_DIRCOPY = "pl-dircopy"
PL_CHRNIST = "pl-chrnist"

# Files live here on the CUBE host and will be exposed by dircopy:
HOST_DIR = "home/chris/chrnist_stuff"  # what you used in the UI

# After dircopy runs, chrNIST sees them with slashes flattened to underscores:
WEIGHTS = "home_chris_chrnist_stuff/best.ckpt"
IMAGE   = "home_chris_chrnist_stuff/custom_4.png"

COMPUTE = "host"
TITLE_DIRCOPY = "stage input"
TITLE_CHRNIST = "predict digits"


async def main_async() -> None:
    async with (await ChrisClient.from_login(
        url=SERVER,
        username=USER,
        password=PASS,
    )) as chris:
        # 1) stage data with pl-dircopy (root node; creates a feed)
        dircopy = await chris.search_plugins(name_exact=PL_DIRCOPY).get_only()
        staging = await dircopy.create_instance(
            title=TITLE_DIRCOPY,
            compute_resource_name=COMPUTE,
            dir=HOST_DIR,  # plugin-specific argument
        )

        # 2) run chrNIST on the staged directory
        chrnist = await chris.search_plugins(name_exact=PL_CHRNIST).get_only()
        await chrnist.create_instance(
            previous=staging,               # chain to dircopy
            title=TITLE_CHRNIST,
            compute_resource_name=COMPUTE,
            mode="predict",                 # plugin-specific args
            weights=WEIGHTS,
            image=IMAGE,
        )
        # Quiet on success.

def main() -> None:
    try:
        asyncio.run(main_async())
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
