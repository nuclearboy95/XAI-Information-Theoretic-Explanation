import argparse
from codes.itattr import AttributionConfig, save_map

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='data/parachute.png')
parser.add_argument('--K', type=int, default=8)
parser.add_argument('--N', type=int, default=8)
parser.add_argument('--S', type=int, default=1)

args = parser.parse_args()


def main():
    config = AttributionConfig(image_path=args.image_path, K=args.K, S=args.S, N=args.N)
    save_map(config)


if __name__ == '__main__':
    main()
