import json


def get_temp_path(path):
    return path.parent / f"{path.stem}_temp.jsonl"


def read_data(path):
    if path.is_dir():
        data = []
        for file in path.glob("*.json"):
            with file.open() as f:
                data.append(json.load(f))
        return data

    if path.is_file():
        if path.suffix == ".jsonl":
            with path.open("r") as f:
                data = []
                for line in f:
                    data.append(json.loads(line))
            return data
        else:
            with path.open("r") as f:
                data = json.load(f)
            return data

    raise ValueError(f"Invalid path: {path}")


def write_data(data, path):
    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    with path.open("w") as f:
        if type(data) == str:
            f.write(data)
        else:
            json.dump(data, f)
