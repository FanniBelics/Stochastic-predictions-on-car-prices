import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    with open("model_results.json", "r") as f:
        results = json.load(f)
        
    actual_prices = results["sample_cars"]["actual_prices"]
    models = results["models"]
    model_names = list(models.keys())
    
    fig = plt.figure(figsize=(18, 10))
    axes = fig.subplot_mosaic([
        ["r2", "mae", "rmse"],
        ["scatter", "scatter", "scatter"]
    ])
    

    model_r2 = [models[name]["r2"] for name in model_names]
    sns.barplot(x=model_names, y=model_r2, ax=axes["r2"], palette="viridis")
    axes["r2"].set_title("R² by model")
    axes["r2"].set_ylabel("R²")
    axes["r2"].set_xticklabels(model_names, rotation=20)
    axes["r2"].axhline(y=0, color="black", linewidth=0.8, linestyle="--", label="0")
    axes["r2"].axhline(y=1, color="gray", linewidth=0.8, linestyle=":", label="1")
    axes["r2"].legend()
    
    model_mae = [models[name]["mae"] for name in model_names]
    sns.barplot(x=model_names, y=model_mae, ax=axes["mae"], palette="viridis")
    axes["mae"].set_title("MAE by model")
    axes["mae"].set_ylabel("MAE")
    axes["mae"].set_xticklabels(model_names, rotation=20)
    axes["mae"].axhline(y=min(model_mae), color="black", linewidth=0.8, linestyle="--", label=f"${min(model_mae):.2f}$")
    axes["mae"].axhline(y=max(model_mae), color="gray", linewidth=0.8, linestyle=":", label=f"${max(model_mae):.2f}$")
    axes["mae"].legend()
    
    model_rmse = [models[name]["rmse"] for name in model_names]
    sns.barplot(x=model_names, y=model_rmse, ax=axes["rmse"], palette="viridis")
    axes["rmse"].set_title("RMSE by model")
    axes["rmse"].set_ylabel("RMSE")
    axes["rmse"].set_xticklabels(model_names, rotation=20)
    axes["rmse"].axhline(y=min(model_rmse), color="black", linewidth=0.8, linestyle="--", label=f"${min(model_rmse):.2f}$")
    axes["rmse"].axhline(y=max(model_rmse), color="gray", linewidth=0.8, linestyle=":", label=f"${max(model_rmse):.2f}$")
    axes["rmse"].legend()
    
    axes["scatter"].plot(range(len(actual_prices)), actual_prices, '-', linewidth=2, label="Actual car prices", zorder=5)
    #colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    colors = [plt.cm.tab10(i) for i in range(len(model_names))]
    
    for i, name in enumerate(model_names):
        preds = models[name]["predictions"]
        lower = models[name]["lower_bound"]
        upper = models[name]["upper_bound"]
        axes["scatter"].plot(range(len(preds)), preds,
                            color=colors[i], linewidth=1, label=name)
        if lower and upper:
            axes["scatter"].fill_between(range(len(preds)), lower, upper,
                                        color=colors[i], alpha=0.15)
            
    axes["scatter"].set_title(f"Predicted vs Actual prices — {len(actual_prices)} sample cars")
    axes["scatter"].set_xlabel("Car index")
    axes["scatter"].set_ylabel("Price ($)")
    axes["scatter"].legend(loc="upper right", fontsize=8)
    axes["scatter"].set_xticks(range(len(actual_prices)))
    axes["scatter"].set_xticklabels(range(1, len(actual_prices) + 1), fontsize=7)
    axes["scatter"].set_xticks(range(len(actual_prices)))
    axes["scatter"].xaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)
    axes["scatter"].set_axisbelow(True)

    
    
    plt.tight_layout(pad=2.0)
    plt.savefig("model_comparison.png", bbox_inches="tight")


    plt.show()


    
if __name__ == "__main__":
    main()