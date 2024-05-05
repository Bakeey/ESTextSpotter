import os
import torch
import pickle
import click

@click.command()
@click.option('--input_dir', type=str, default='/cluster/home/kiten/input/test', help='Path to the input directory containing pickle files')
@click.option('--output_dir', type=str, default='/cluster/home/kiten/output/test/text_detections', help='Path to the output directory')
def main(input_dir, output_dir):
    # Check if the output directory exists, if not, create it
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.pkl'):
            file_path = os.path.join(input_dir, filename)

            with open(file_path, 'rb') as file:
                # Load data from the file using pickle
                data = pickle.load(file)

            # Move all tensors to CPU before serializing
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    data[key] = value.cpu()
            for i, s in enumerate(data.get('all_scores', [])):
                if isinstance(s, torch.Tensor):
                    data['all_scores'][i] = s.cpu()
            for i, s in enumerate(data.get('scores', [])):
                if isinstance(s, torch.Tensor):
                    data['scores'][i] = s.cpu()

            # Replace the current extension with .pkl
            base_name, _ = os.path.splitext(filename)
            output_file_path = os.path.join(output_dir, base_name + '_cpu.pkl')
            with open(output_file_path, 'wb') as f:
                pickle.dump(data, f)

if __name__ == '__main__':
    main()
