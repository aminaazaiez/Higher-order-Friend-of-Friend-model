from pathlib import Path
dirpath = Path("./out/simulations/ho_fof_resampled_T")  # folder containing the files
names_to_delete = [
    'n-30000_M-{2: 7, 3: 2}_p-0.000_n0-20_multiedges-True_seed-9.joblib',
    'n-30000_M-{2: 8}_p-0.000_n0-20_multiedges-True_seed-8.joblib',



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