import yaml
from pathlib import Path
from datetime import datetime

from gerry.data.map_pack import load_map_pack
from gerry.algos.kmeans_softcap import kmeans_softcap_labels
from scripts.export_run import export_run

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    pack_dir = cfg["data"]["map_pack_dir"]
    out_root = Path(cfg["output"]["out_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    num_districts = int(cfg["run"].get("num_districts", 17))
    pop_tolerance = float(cfg["run"].get("pop_tolerance", 0.05))
    max_iter = int(cfg["run"].get("max_iter", 80))
    alpha = float(cfg["run"].get("alpha", 10000))
    seed = int(cfg["run"].get("seed", 42))

    pack = load_map_pack(pack_dir)

    labels = kmeans_softcap_labels(
        coords=pack.coords,
        weight=pack.weight,
        num_districts=num_districts,
        pop_tolerance=pop_tolerance,
        max_iter=max_iter,
        alpha=alpha,
        seed=seed,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"kmeans_seed{seed}_{run_id}"
    export_run(pack_dir=pack_dir, pack=pack, labels=labels, run_dir=run_dir, title="KMeans (soft pop constraint)")

    print("Saved:", run_dir)

if __name__ == "__main__":
    main()
