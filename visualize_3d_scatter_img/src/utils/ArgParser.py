import argparse    # 1. argparseをインポート


class ArgParser:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description='jsonを設定してください')
        parser.add_argument('--json', default='../data/json/settings.json')
        args = parser.parse_args()
        self.json_file = args.json

    def get_json_file(self):
        return self.json_file
