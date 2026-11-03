# Data sources

No images are committed to this repository. They are created or downloaded by scripts into `data/`, which git ignores.

## In use

| Dataset | What it is | License | Used for | How to get it |
|---|---|---|---|---|
| Rendered boards (this project) | Random positions drawn with `board2fen/render.py` and `scene.py`, using fonts installed on the computer | Our own code ([MIT](LICENSE)) | Training, testing | `scripts/generate_dataset.py` |
| [samryan18/chess-dataset](https://github.com/samryan18/chess-dataset) | 500 real photos, 6000 × 4000 pixels, taken from above: one vinyl board and one set of 3D pieces. The FEN is the file name. The set also has its own straightened copies of the photos. | [MIT](https://github.com/samryan18/chess-dataset/blob/master/LICENSE.md): copyright 2019 Samuel Ryan, Mukund Venkateswaran, Kurt Convey, Michael Deng | Measuring real-photo accuracy | `scripts/download_data.py` (pinned to commit `62a8f5b`, about 2.2 GB) |

## Considered, not in use

| Dataset | What it is | Why not (yet) |
|---|---|---|
| [ChessReD](https://github.com/ThanosM97/end-to-end-chess-recognition) | 10,800 smartphone photos of 100 games, from many angles. Hosted at [4TU.ResearchData](https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f). | Much larger download. A good test of other boards, sets and camera angles for later. |
| [Roboflow "Chess Pieces"](https://public.roboflow.com/object-detection/chess-full) | About 290 photos with a bounding box per piece | Labels are boxes, not positions, so they'd have to be converted. |
| [Kaggle "Chess Positions"](https://www.kaggle.com/datasets/koryakinp/chess-positions) | Rendered 2D boards with the FEN as the file name | Not photos, and we render our own 2D boards already. |

Check a dataset's license on its own page before using it.
