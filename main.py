#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Приложение для решения систем линейных алгебраических уравнений методом Гаусса.

Точка входа в приложение.
Запуск: python main.py
"""

import sys
from pathlib import Path

# Добавляем текущую директорию в путь для импорта модулей
sys.path.insert(0, str(Path(__file__).parent))

from main_window import main

if __name__ == "__main__":
    main()
