"""
Minesweeper with ai confidence overlay
Left-click : reveal
Right-click : flag / unflag
"""

import tkinter as tk
from tkinter import messagebox
import random
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
import login

#db setup
load_dotenv()
_DB_CONNECTED = False

#make sure that db is connected
def _ensure_db():
    global _DB_CONNECTED
    if _DB_CONNECTED:
        return True
    username = os.getenv("DB_USERNAME")
    password = os.getenv("DB_PASSWORD")
    schema   = os.getenv("DB_SCHEMA")
    success, error = login.login(username, password, schema)
    if not success:
        print(f"DB login failed: {error} — stats will not be saved.")
        return False
    _DB_CONNECTED = True
    return True

#insert a new row in model_stats with the values from the game
def save_game_stats(model_name, won, total_guesses, correct_guesses):
    if not _ensure_db():
        return
    schema = os.getenv("DB_SCHEMA")
    db = login.db()
    if db is None:
        return
    try:
        #query to select something ez and make sure that it exists
        check_q = f"SELECT 1 FROM {schema}.model_statistics WHERE model_name = %s;"
        with db.cursor() as c:
            c.execute(check_q, (model_name,))
            exists = c.fetchone() is not None

        if exists:
            #update if it already exists
            update_q = f"""
                UPDATE {schema}.model_statistics
                SET games_played    = games_played    + 1,
                    games_won       = games_won       + %s,
                    total_guesses   = total_guesses   + %s,
                    correct_guesses = correct_guesses + %s
                WHERE model_name = %s;
                """
            with db.cursor() as c:
                c.execute(update_q, (int(won), total_guesses, correct_guesses, model_name))
        else:
            #make a new row if its a new model
            insert_q = f"""
                INSERT INTO {schema}.model_statistics
                (model_name, games_played, games_won, total_guesses, correct_guesses)
                VALUES (%s, 1, %s, %s, %s);
                """
            with db.cursor() as c:
                c.execute(insert_q, (model_name, int(won), total_guesses, correct_guesses))

        db.commit()
    except Exception as e:
        db.rollback()

#size and #of mines presets
PRESETS = {
    "Beginner":     (9,  9,  10),
    "Intermediate": (16, 16, 40),
    "Expert":       (16, 30, 99),
}

#colors of #mines clues
CLUE_COLORS = {
    1: "#1565C0", 2: "#2E7D32", 3: "#C62828",
    4: "#283593", 5: "#6D1A36", 6: "#00695C",
    7: "#212121", 8: "#757575",
}

#unrevealed cell color
CELL_BG = "#BBBBBB"

#model loading - scans the given dir for .pth files and returns a list of (name, params) tuples
def load_models(model_dir="../models"):
    models = []
    paths = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
    for p in paths:
        try:
            params = torch.load(p, map_location="cpu", weights_only=False)
            name = os.path.basename(p)
            models.append((name, params))
        except Exception as e:
            print(f"failed to load {p}: {e}")
    return models

#inference
#takes the cell (tr, tc), builds the 5x5 neighborhood feature vec (w/o the center)
#to match the same databse encoding
#0-8 : revealed clue
# -1 : hidden/unrevealed
# -2 : border/wall
def build_cell_features(game, tr, tc):
    cells = []
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if dr == 0 and dc == 0:
                #skip the center
                continue
            r, c = tr + dr, tc + dc #translate to the actual coords
            if not (0 <= r < game.rows and 0 <= c < game.cols):
                cells.append(-2) #border
            elif game.shown[r][c]:
                cells.append(game.clues[r][c]) #revealed clue 0-8
            else:
                cells.append(-1) #hidden
    return cells  #should be of length 24

#run a forward pass through the cnn and returns the confidence val
def run_inference(params, cells_24, global_density):
    """
    Run a forward pass through the CNN defined in train-model.py.
    Returns a float in [0, 1] representing P(safe).
    """
    cells = list(cells_24)
    cells.insert(12, -1)

    cells_shifted = torch.tensor([v + 2 for v in cells], dtype=torch.long)
    grid_onehot   = F.one_hot(cells_shifted, num_classes=11).float()
    grid_2d       = grid_onehot.view(5, 5, 11).permute(2, 0, 1).unsqueeze(0)
    density       = torch.tensor([[global_density]], dtype=torch.float32)

    with torch.no_grad():
        x = F.conv2d(grid_2d, params["conv1_w"], params["conv1_b"])
        x = F.relu(x)
        x = F.conv2d(x, params["conv2_w"], params["conv2_b"])
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, density), dim=1)
        x = F.relu(F.linear(x, params["fc1_w"], params["fc1_b"]))
        x = F.linear(x, params["fc2_w"], params["fc2_b"])
        return torch.sigmoid(x).item()

#helper func -- check to see if the cell has a revealed neighbor
def has_revealed_neighbor(game, tr, tc):
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:
                continue
            r, c = tr + dr, tc + dc
            if 0 <= r < game.rows and 0 <= c < game.cols and game.shown[r][c]:
                return True
    return False

#for every unrevealed, unflagged cell computeP(safe)
#returns a 2d list of either floats or None
def compute_confidence_grid(game, params):
    if not game.started:
        return None

    hidden_count = sum(
        1 for r in range(game.rows) for c in range(game.cols)
        if not game.shown[r][c] and not game.flagged[r][c]
    )
    mines_left     = game.n_mines - game.flags_placed()
    global_density = mines_left / hidden_count if hidden_count > 0 else 0.0

    grid = [[None] * game.cols for _ in range(game.rows)]
    for r in range(game.rows):
        for c in range(game.cols):
            if not game.shown[r][c] and not game.flagged[r][c]:
                if has_revealed_neighbor(game, r, c):
                    cells      = build_cell_features(game, r, c)
                    grid[r][c] = run_inference(params, cells, global_density)
    return grid


# color mapping
# 0.0 - red - #E53935
# 0.5 - grey - #BBBBBB
# 1.0 - green - #43A047
def confidence_to_color(p_safe):
    if p_safe >= 0.5:
        #grey to green
        t = (p_safe - 0.5) * 2
        r = int(0xBB + t * (0x43 - 0xBB))
        g = int(0xBB + t * (0xA0 - 0xBB))
        b = int(0xBB + t * (0x47 - 0xBB))
    else:
        #red to grey
        t = p_safe * 2
        r = int(0xE5 + t * (0xBB - 0xE5))
        g = int(0x39 + t * (0xBB - 0x39))
        b = int(0x35 + t * (0xBB - 0x35))
    return f"#{r:02X}{g:02X}{b:02X}"


class Game:
    def __init__(self, rows, cols, n_mines):
        self.rows    = rows
        self.cols    = cols
        self.n_mines = n_mines
        self.mines   = [[False]*cols for _ in range(rows)]
        self.clues   = [[0]*cols     for _ in range(rows)]
        self.shown   = [[False]*cols for _ in range(rows)]
        self.flagged = [[False]*cols for _ in range(rows)]
        self.started = False
        self.over    = False
        self.won     = False

    def _in(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _nb(self, r, c):
        return [(r+dr, c+dc)
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr or dc) and self._in(r+dr, c+dc)]

    def _setup(self, sr, sc):
        safe = {(sr+dr, sc+dc)
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if self._in(sr+dr, sc+dc)}
        safe.add((sr, sc))
        pool = [(r, c) for r in range(self.rows)
                       for c in range(self.cols)
                       if (r, c) not in safe]
        for r, c in random.sample(pool, min(self.n_mines, len(pool))):
            self.mines[r][c] = True
        for r in range(self.rows):
            for c in range(self.cols):
                if not self.mines[r][c]:
                    self.clues[r][c] = sum(
                        self.mines[nr][nc] for nr, nc in self._nb(r, c))
        self.started = True

    def reveal(self, r, c):
        if self.over or self.flagged[r][c] or self.shown[r][c]:
            return
        if not self.started:
            self._setup(r, c)
        if self.mines[r][c]:
            self.shown[r][c] = True
            self.over = True
            return
        self._flood(r, c)
        if all(self.shown[r][c] or self.mines[r][c]
               for r in range(self.rows)
               for c in range(self.cols)):
            self.won = self.over = True

    def _flood(self, r, c):
        stack = [(r, c)]
        while stack:
            r, c = stack.pop()
            if not self._in(r, c) or self.shown[r][c] or self.flagged[r][c]:
                continue
            self.shown[r][c] = True
            if self.clues[r][c] == 0:
                stack.extend(self._nb(r, c))

    def flag(self, r, c):
        if not self.over and not self.shown[r][c]:
            self.flagged[r][c] = not self.flagged[r][c]

    def flags_placed(self):
        return sum(self.flagged[r][c]
                   for r in range(self.rows) for c in range(self.cols))

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Minesweeper + AI Confidence")
        self.resizable(False, False)

        self.preset     = tk.StringVar(value="Intermediate")
        self.btns       = []
        self.game       = None
        self.grid_frame = None
        self.conf_grid  = None
        self.show_conf  = tk.BooleanVar(value=True)

        self.models    = load_models()
        self.model_idx = 0

        # Per-game guess tracking.
        self._total_guesses   = 0
        self._correct_guesses = 0

        self._build_top()
        self._build_model_panel()
        self.bind("<F2>", lambda _: self.new_game())
        self.new_game()

    #ui stuff
    def _build_top(self):
        top = tk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        for p in PRESETS:
            tk.Radiobutton(top, text=p, variable=self.preset,
                           value=p, command=self.new_game).pack(side=tk.LEFT)

        self.mine_lbl = tk.Label(top, text="", width=12)
        self.mine_lbl.pack(side=tk.RIGHT)
        tk.Button(top, text="New Game (F2)",
                  command=self.new_game).pack(side=tk.RIGHT, padx=6)
        tk.Checkbutton(top, text="Show AI", variable=self.show_conf,
                       command=self._redraw).pack(side=tk.RIGHT, padx=4)

    #slider to pick between the models
    def _build_model_panel(self):
        panel = tk.Frame(self, bd=1, relief=tk.GROOVE)
        panel.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 4))

        tk.Label(panel, text="Model:", font=("Courier", 10, "bold")).pack(
            side=tk.LEFT, padx=(6, 2), pady=4)

        if not self.models:
            tk.Label(panel, text="No *.pth files found in ../models/",
                     fg="red").pack(side=tk.LEFT)
            return

        #slider — one tick per model
        self.model_slider = tk.Scale(
            panel,
            from_=0, to=max(0, len(self.models) - 1),
            orient=tk.HORIZONTAL,
            showvalue=False,
            length=220,
            command=self._on_slider,
        )
        self.model_slider.pack(side=tk.LEFT, padx=4, pady=2)

        #label showing the currently selected model filename
        self.model_name_lbl = tk.Label(
            panel,
            text=self.models[0][0],
            font=("Courier", 9),
            anchor="w",
            width=46,
        )
        self.model_name_lbl.pack(side=tk.LEFT, padx=6)

    def _on_slider(self, val):
        self.model_idx = int(val)
        if self.models:
            self.model_name_lbl.config(text=self.models[self.model_idx][0])
        self._run_inference_and_redraw()

    #game flow
    def new_game(self):
        if self.grid_frame is not None:
            self.grid_frame.destroy()

        self.grid_frame = tk.Frame(self, bd=2, relief=tk.SUNKEN)
        self.grid_frame.pack(padx=6, pady=6)

        self.btns      = []
        self.conf_grid = None
        self._total_guesses   = 0
        self._correct_guesses = 0

        rows, cols, mines = PRESETS[self.preset.get()]
        self.game = Game(rows, cols, mines)

        for r in range(rows):
            row_btns = []
            for c in range(cols):
                b = tk.Label(
                    self.grid_frame,
                    width=2, height=1,
                    font=("Courier", 11, "bold"),
                    relief=tk.RAISED, bd=2,
                    bg=CELL_BG, text="",
                )
                b.grid(row=r, column=c, padx=0, pady=0)
                b.bind("<Button-1>", lambda e, r=r, c=c: self._left(r, c))
                b.bind("<Button-2>", lambda e, r=r, c=c: self._right(r, c))
                b.bind("<Button-3>", lambda e, r=r, c=c: self._right(r, c))
                row_btns.append(b)
            self.btns.append(row_btns)

        self._update_label()

    def _left(self, r, c):
        g = self.game
        if g.over or g.flagged[r][c] or g.shown[r][c]:
            return

        #count as a guess only when there is no revealed neighbor
        if not has_revealed_neighbor(g, r, c):
            self._total_guesses += 1
            if not g.mines[r][c]:
                self._correct_guesses += 1

        g.reveal(r, c)
        self._run_inference_and_redraw()
        if g.over:
            self._end()

    def _right(self, r, c):
        if self.game.over:
            return
        self.game.flag(r, c)
        self._run_inference_and_redraw()

    def _run_inference_and_redraw(self):
        if self.models and self.game.started and not self.game.over:
            _, params      = self.models[self.model_idx]
            self.conf_grid = compute_confidence_grid(self.game, params)
        else:
            self.conf_grid = None
        self._redraw()

    def _redraw(self):
        g = self.game
        for r in range(g.rows):
            for c in range(g.cols):
                b = self.btns[r][c]
                if g.shown[r][c]:
                    if g.mines[r][c]:
                        b.config(text="*", bg="#E53935",
                                 fg="white", relief=tk.SUNKEN)
                    else:
                        n = g.clues[r][c]
                        b.config(
                            text=str(n) if n else "",
                            bg="#D9D9D9",
                            fg=CLUE_COLORS.get(n, "#000000"),
                            relief=tk.SUNKEN,
                        )
                elif g.flagged[r][c]:
                    b.config(text="F", bg=CELL_BG,
                             fg="#E53935", relief=tk.RAISED)
                else:
                    # Unrevealed — apply confidence color if overlay is on
                    bg = CELL_BG
                    if (self.show_conf.get()
                            and self.conf_grid is not None
                            and self.conf_grid[r][c] is not None):
                        bg = confidence_to_color(self.conf_grid[r][c])
                    b.config(text="", bg=bg, fg="black", relief=tk.RAISED)

        self._update_label()

    def _update_label(self):
        self.mine_lbl.config(
            text=f"Mines: {self.game.n_mines - self.game.flags_placed()}")

    def _end(self):
        g = self.game

        #reveal all mines for display
        for r in range(g.rows):
            for c in range(g.cols):
                if g.mines[r][c]:
                    g.shown[r][c] = True
        self.conf_grid = None
        self._redraw()

        #save stats using whichever model is currently active
        if self.models:
            model_name = self.models[self.model_idx][0]
            save_game_stats( model_name, won = g.won, total_guesses = self._total_guesses, correct_guesses = self._correct_guesses,)

        if g.won:
            messagebox.showinfo("Minesweeper", "Win!")
        else:
            messagebox.showinfo("Minesweeper", "BOOM!")


if __name__ == "__main__":
    App().mainloop()