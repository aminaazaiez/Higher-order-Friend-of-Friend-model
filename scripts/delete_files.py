from pathlib import Path
dirpath = Path("./out/simulations/polyadic_closure")  # folder containing the files
names_to_delete = [
    'n-30000_M-{2: 8, 4: 1}_p-0.972_n0-20_H0-random_multiedges-True_seed-34.joblib',
    'n-30000_M-{2: 8, 4: 1}_p-0.972_n0-20_H0-random_multiedges-True_seed-33.joblib',
    'n-30000_M-{2: 8}_p-0.972_n0-20_H0-random_multiedges-True_seed-44.joblib',
    'n-30000_M-{2: 8}_p-0.972_n0-20_H0-random_multiedges-True_seed-43.joblib',


]

# Dry-run: show what would be deleted
to_delete = [p for p in dirpath.iterdir() if p.name in names_to_delete]
for p in to_delete:
    print("Would delete:", p)

# Actually delete
confirm = input("Delete these files? [y/N] ").lower().startswith("y")
if confirm:
    for p in to_delete:
        try:
            p.unlink()
            print("Deleted:", p)
        except FileNotFoundError:
            pass