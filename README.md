# Forest Coin Hunt

A small first-person WebGL game: wander through a low-poly forest and find all
15 gold coins as fast as you can. Written in plain WebGL with no libraries or
build step.

## Play

Open `index.html` in a browser (double-clicking it works), or serve the folder:

```
python3 -m http.server
```

then visit http://localhost:8000.

## Controls

- **W A S D** / arrow keys: move (left/right arrows turn)
- **Mouse**: look around (click the game to capture the mouse, or click and drag)
- **Shift**: run
- **Space**: jump
- **Esc**: pause

The "coin sense" under the counter tells you how close the nearest coin is:
Cold → Cool → Warm → Very hot!
