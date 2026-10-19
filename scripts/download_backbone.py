"""Download the official ImageNet PVTv2-B4 checkpoint on the user's machine."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

URL = 'https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b4.pth'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='pretrained/pvt_v2_b4.pth')
    parser.add_argument('--sha256', help='Optional expected SHA256 obtained from a trusted source')
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError('Already exists: {}'.format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.download')
    digest = hashlib.sha256()
    try:
        request = urllib.request.Request(URL, headers={'User-Agent': 'IVC-reproduction'})
        with urllib.request.urlopen(request, timeout=60) as source, temporary.open('wb') as output:
            while True:
                block = source.read(1024*1024)
                if not block:
                    break
                output.write(block)
                digest.update(block)
        if args.sha256 and digest.hexdigest().lower() != args.sha256.lower():
            raise ValueError('SHA256 mismatch: {}'.format(digest.hexdigest()))
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    print('Saved {}\nSHA256 {}'.format(target, digest.hexdigest()))


if __name__ == '__main__':
    main()
