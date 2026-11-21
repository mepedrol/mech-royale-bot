# Mech Royale bot ("survivor")

A mechfight/1 bot for the Builders Mech Royale track. Python standard library only.

- `bot.py`: decision logic (zone timing, hit-and-hide duels, finishing weak enemies, regen, loot, supply drops, upgrades, aiming with lead).
- `navmap.py`: 2 m walk grid built from `/arena/v2` (terrain + prisms), line of sight, A* and a distance field to the safe spot.
- `mechfight.py`: the kit's stdin/stdout wrapper (unchanged).

Build and check (from the kit folder):

```sh
docker build -t my-bot .
python3 check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/arena:/arena:ro" my-bot
```
