#!/usr/bin/env python3

"""
(C) 2026 Jack Lloyd

Botan is released under the Simplified BSD License (see license.txt)
"""

import argparse
import configparser
import hashlib
import os
import subprocess
import sys
import tempfile
import urllib.request

def load_config(dep_name):
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               '..', '..', 'configs', 'ci_deps.conf')
    config = configparser.ConfigParser()
    config.read(config_path)

    if dep_name not in config:
        print("Unknown dependency %s - available options are %s" % (
            dep_name, ','.join(config.sections())))
        sys.exit(1)

    section = config[dep_name]
    url = section.get('url')
    sha256 = section.get('sha256')
    if not url or not sha256:
        print("Bad config entry for %s" % (dep_name))
        sys.exit(1)

    return url, sha256

def download(url, max_mb):
    max_bytes = max_mb * 1024 * 1024
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        content_length = resp.headers.get('Content-Length')

        content_length = int(content_length) if content_length is not None else None

        if (content_length is not None) and (content_length > max_bytes):
            print("Download of %s too large, server reports %d bytes" % (url, content_length))
            sys.exit(1)

        chunks = []
        total = 0
        while True:
            chunk = resp.read(256 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if (content_length is not None) and (total > content_length):
                print("Server sent too much data for %s, reported %d" % (url, content_length))
                sys.exit(1)
            if total > max_bytes:
                print("Server sent too much data for %s" % (url))
                sys.exit(1)
            chunks.append(chunk)

    return b''.join(chunks)

def main():
    parser = argparse.ArgumentParser(description='Download a CI dependency with integrity verification')
    parser.add_argument('dep_name', help='Dependency name (section in ci_deps.conf)')
    parser.add_argument('output_path', nargs='?', default=None,
                        help='Output file path (default: filename from URL in current directory)')
    parser.add_argument('--max-download-mb', default=48, type=int,
                        help='Maximum download size in MB')
    parser.add_argument('--extract', default=None, metavar='CMD',
                        help='Extract after download using CMD template (eg "tar -xf {file}")')
    args = parser.parse_args()

    url, expected_sha256 = load_config(args.dep_name)

    data = download(url, args.max_download_mb)

    computed_sha256 = hashlib.sha256(data).hexdigest()
    if computed_sha256 != expected_sha256:
        print("Checksum failure downloading %s - got %s (%d bytes)" % (
            url, computed_sha256, len(data)))
        return 1

    if args.extract:
        filename = os.path.basename(urllib.request.url2pathname(url.split('?')[0]))
        with tempfile.NamedTemporaryFile(prefix='ci_dep_', suffix='_' + filename, delete=False) as f:
            f.write(data)
            tmp_path = f.name

        cmd = args.extract.replace('{file}', tmp_path)
        try:
            subprocess.run(cmd, shell=True, check=True)
        finally:
            os.unlink(tmp_path)
    else:
        if args.output_path:
            output = args.output_path
        else:
            output = os.path.basename(urllib.request.url2pathname(url.split('?')[0]))

        with open(output, 'wb') as f:
            f.write(data)

        print(output)

    return 0

if __name__ == '__main__':
    sys.exit(main())
