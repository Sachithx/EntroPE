from data_provider.data_factory import data_provider


args = {"data": "custom", 
        "embed": "timeF", 
        "batch_size": 128,
        "freq": "h",
        "features": "M",
        "target": "OT",
        "root_path": "dataset/",
        "data_path": "weather.csv",
        "seq_len": 96, 
        "label_len": 48, 
        "pred_len": 24,
        "num_workers": 0
        }

def get_data(args, flag):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader

if __name__ == "__main__":
    args = type('Args', (object,), args)  # Convert dict to object for easier access
    data_set, data_loader = get_data(args, 'test')
    print(f"Data set length: {len(data_set)}")
    for batch in data_loader:
        print(batch)
        break  # Just print the first batch for demonstration
    print("Data loading complete.")