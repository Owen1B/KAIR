import os
import argparse
import csv
import json
import glob
import matplotlib.pyplot as plt
import wandb


def _plot_scatter(x_data, y_data, c_data, x_label, y_label, c_label, title, plot_path):
    """Helper function to create a scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(x_data, y_data, alpha=0.7, c=c_data, cmap='viridis')

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(c_label)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def replot_from_csv(exp_dir):
    """
    Reads metrics from a CSV file within the experiment directory,
    re-plots them, and logs to a wandb run.
    """
    log_dir = os.path.join(exp_dir)
    csv_path = os.path.join(log_dir, 'metrics_correlation.csv')

    if not os.path.isfile(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    # 1. Read all historical data from CSV
    steps, test_lpips_data, test_ssim_data, test_psnr_data, val_lpips_data = [], [], [], [], []
    try:
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Skip header
            for row in reader:
                steps.append(int(row[0]))
                test_lpips_data.append(float(row[1]))
                test_ssim_data.append(float(row[2]))
                test_psnr_data.append(float(row[3]))
                val_lpips_data.append(float(row[4]))
    except (IOError, IndexError, ValueError) as e:
        print(f"Error reading or parsing CSV file: {e}")
        return

    if not steps:
        print("No data to plot in CSV file.")
        return

    # 2. Find and load option file to get wandb config
    option_file = None
    options_dir = os.path.join(exp_dir, 'options')
    if os.path.isdir(options_dir):
        json_files = glob.glob(os.path.join(options_dir, '*.json'))
        if json_files:
            option_file = json_files[0]  # Use the first json file found

    if not option_file:
        print(f"Error: No option file (.json) found in {options_dir}")
        return

    with open(option_file, 'r') as f:
        opt = json.load(f)

    # 3. Initialize wandb
    run_id = None
    run_id_path = os.path.join(log_dir, 'wandb_run_id.json')
    if os.path.exists(run_id_path):
        with open(run_id_path, 'r') as f:
            run_id = json.load(f).get('run_id')

    try:
        wandb.init(
            entity=opt.get('wandb', {}).get('entity'),
            project=opt.get('wandb', {}).get('project'),
            name=opt.get('task', 'task'),
            group=opt.get('wandb', {}).get('group'),
            config=opt,
            id=run_id,
            resume="allow" if run_id else None
        )
    except Exception as e:
        print(f"Error initializing wandb: {e}")
        print("Please ensure you are logged into wandb.")
        return

    # 4. Plot and log to wandb
    plots = {
        'LPIPS_Correlation': {
            'x_data': test_lpips_data, 'y_data': val_lpips_data,
            'x_label': 'test/lpips_local (no noise)', 'y_label': 'val_2_poisson/lpips_local (with noise)',
            'path': os.path.join(log_dir, 'correlation_lpips.png')
        },
        'SSIM_vs_LPIPS_Correlation': {
            'x_data': test_ssim_data, 'y_data': val_lpips_data,
            'x_label': 'test/ssim_local (no noise)', 'y_label': 'val_2_poisson/lpips_local (with noise)',
            'path': os.path.join(log_dir, 'correlation_ssim_vs_lpips.png')
        },
        'PSNR_vs_LPIPS_Correlation': {
            'x_data': test_psnr_data, 'y_data': val_lpips_data,
            'x_label': 'test/psnr_local (no noise)', 'y_label': 'val_2_poisson/lpips_local (with noise)',
            'path': os.path.join(log_dir, 'correlation_psnr_vs_lpips.png')
        }
    }

    wandb_images = {}
    for title, data in plots.items():
        plot_path = _plot_scatter(
            x_data=data['x_data'], y_data=data['y_data'], c_data=steps,
            x_label=data['x_label'], y_label=data['y_label'], c_label='Iteration Step',
            title=title.replace('_', ' '), plot_path=data['path']
        )
        wandb_images[f'correlation_plots/{title}'] = wandb.Image(plot_path)

    last_step = steps[-1] if steps else 0
    wandb.log({**wandb_images, 'iteration': last_step})

    print(f"Plots have been re-generated and logged to wandb run: {wandb.run.url}")
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Re-plot correlation charts from a training run and upload to wandb.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--exp_dir',
        type=str,
        required=True,
        help="Path to the experiment directory.\nThis directory should contain a 'log' folder with 'metrics_correlation.csv'."
    )
    args = parser.parse_args()
    replot_from_csv(args.exp_dir)


if __name__ == '__main__':
    main() 