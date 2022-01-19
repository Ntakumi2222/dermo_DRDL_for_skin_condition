import argparse


class ReconstructionOptions:
    def __init__(self):
        parser = argparse.ArgumentParser(description='初期値設定')

        parser.add_argument('--DIR', type=str, default='', help='DIRの絶対パス')

        args = parser.parse_args()
        self.DIR = args.DIR

    def disp_info(self):
        print(f'Input Data Info')
        print(f'---------------------------------------------')
        print(f'    DIR:{self.DIR}')
        print(f'---------------------------------------------')
        pass
