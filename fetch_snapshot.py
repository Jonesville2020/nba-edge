"""Download this season's data once and save it to data/ for the app to read."""
import pathlib
import data

out = pathlib.Path("data")
out.mkdir(exist_ok=True)
logs, bio, team_adv, player_adv = data.load_player_data()
logs.to_csv(out / "logs.csv", index=False)
bio.to_csv(out / "bio.csv", index=False)
team_adv.to_csv(out / "team_adv.csv", index=False)
player_adv.to_csv(out / "player_adv.csv", index=False)
print("saved", len(logs), "player-games through", logs.date.max())
