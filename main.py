#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import logging
import sys
from typing import Optional, Sequence

from src.cli import parse_arguments
from src.config import LOG_DATE_FORMAT, LOG_FORMAT
from src.ocr_selftest import run_ocr_selftest
from src.video_runner import run_video

LOGGER = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Konsol loglamasını yapılandırır."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        stream=sys.stdout,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Uygulamanın giriş noktası."""
    args = parse_arguments(argv)
    setup_logging(args.verbose)

    
    if args.test_ocr:
        return run_ocr_selftest(args.test_ocr)

    if not args.source:
        LOGGER.error("--source zorunludur (veya tani icin --test-ocr kullanin).")
        return 1

    return run_video(args)


if __name__ == "__main__":
    sys.exit(main())
