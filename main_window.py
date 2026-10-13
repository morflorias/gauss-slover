"""
Главное окно приложения для решения СЛАУ методом Гаусса.
Разработано с использованием PyQt5 для платформы Windows Desktop.

Обновлено: строгий и наглядный UI для пошагового вывода, плоская тема.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Optional, List
import re
import ctypes
import platform
import openpyxl

from openpyxl.styles import Font

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QGroupBox, QTextEdit, QFileDialog,
    QMessageBox, QSplitter, QFrame, QHeaderView, QComboBox,
    QDoubleSpinBox, QStatusBar, QToolBar, QAction, QTextBrowser,
    QScrollArea, QSizePolicy, QSpacerItem, QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt, QTimer, QUrl, QSize
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor, QTextCursor, QCursor
from PyQt5.QtCore import pyqtSignal

from gauss_solver import GaussSolver, SolutionStatus, SolutionResult


# ============================================================================
# CSS СТИЛИ ДЛЯ HTML ФОРМАТИРОВАНИЯ
# ============================================================================

HTML_STYLES = """
<style>
    body { 
        font-family: 'Segoe UI', Arial, sans-serif; 
        background-color: #2d2d2d; 
        color: #e0e0e0; 
        font-size: 13px; 
        line-height: 1.5; 
        margin: 10px; 
    }
    .section-header { 
        font-size: 14px; 
        font-weight: bold; 
        color: #5c9edd; 
        margin: 20px 0 10px 0; 
        border-bottom: 1px solid #454545; 
        padding-bottom: 4px; 
        text-transform: uppercase; 
        letter-spacing: 0.5px; 
    }
    .step-header { 
        font-weight: bold; 
        color: #ffffff; 
        margin: 15px 0 8px 0; 
        background-color: #363636; 
        padding: 4px 8px; 
        border-radius: 4px; 
        display: inline-block;
    }
    .equation { 
        font-family: 'Consolas', monospace; 
        color: #b0b0b0; 
        margin-left: 10px; 
        margin-bottom: 4px; 
    }
    .solution-line { 
        font-family: 'Consolas', monospace; 
        color: #4caf50; 
        margin-left: 10px; 
        font-size: 14px; 
        margin-bottom: 4px; 
    }
    .matrix-table { 
        background-color: #323232; 
        border: 1px solid #454545; 
        margin: 12px 0 15px 10px; 
        font-family: 'Consolas', monospace; 
        font-size: 13px; 
    }
    .matrix-table td { 
        text-align: right; 
        padding: 6px 14px; 
    }
    .matrix-table td.b-column { 
        border-left: 1px solid #555555; 
        color: #9c7cfa; 
    }
    .zero { 
        color: #666666; 
    }
    .text-line { 
        margin: 6px 0; 
    }
    .error-text { 
        color: #ef5350; 
        font-weight: bold; 
    }
    .warning-text { 
        color: #ff9800; 
        font-weight: bold; 
    }
</style>
"""


class StepsFormatter:
    """
    Класс для строгого форматирования пошагового решения в HTML.
    """

    def format_steps(self, steps: List[str]) -> str:
        """Форматирование списка шагов в чистый HTML."""
        html_parts = [HTML_STYLES, "<body>"]

        i = 0
        while i < len(steps):
            step = steps[i].strip()
            if not step:
                i += 1
                continue

            if step in["ПРЯМОЙ ХОД", "ОБРАТНЫЙ ХОД И АНАЛИЗ", "Исходная расширенная матрица:"]:
                html_parts.append(f'<div class="section-header">{step}</div>')
            elif step.startswith("Шаг"):
                html_parts.append(f'<div class="step-header">{step}</div>')
            elif step.startswith("R") and ("=" in step or "↔" in step):
                html_parts.append(f'<div class="equation">{step}</div>')
            elif step.startswith("x["):
                html_parts.append(f'<div class="solution-line">{step}</div>')
            elif step.startswith("Противоречие") or step == "Система несовместна":
                html_parts.append(f'<div class="text-line error-text">{step}</div>')
            elif "Бесконечное множество" in step:
                html_parts.append(f'<div class="text-line warning-text">{step}</div>')
            elif step.startswith("(") and step.endswith(")"):
                # Собираем все строки матрицы
                matrix_lines = [step]
                j = i + 1
                while j < len(steps) and steps[j].strip().startswith("(") and steps[j].strip().endswith(")"):
                    matrix_lines.append(steps[j].strip())
                    j += 1
                html_parts.append(self._format_matrix(matrix_lines))
                i = j - 1
            else:
                html_parts.append(f'<div class="text-line">{step}</div>')

            i += 1

        html_parts.append("</body>")
        return "\n".join(html_parts)

    def _format_matrix(self, lines: List[str]) -> str:
        """Построение аккуратной таблицы для матрицы."""
        html = ['<table class="matrix-table" cellspacing="0" cellpadding="0">']
        for line in lines:
            clean = line.strip()[1:-1].strip()  # Убираем круглые скобки
            html.append('<tr>')
            if "|" in clean:
                left, right = clean.split("|")
                for val in left.split():
                    v = float(val)
                    cls = "zero" if abs(v) < 1e-10 else ""
                    html.append(f'<td class="{cls}">{v:.4f}</td>')

                v = float(right.strip())
                cls = "zero" if abs(v) < 1e-10 else ""
                html.append(f'<td class="b-column {cls}">{v:.4f}</td>')
            else:
                for val in clean.split():
                    v = float(val)
                    cls = "zero" if abs(v) < 1e-10 else ""
                    html.append(f'<td class="{cls}">{v:.4f}</td>')
            html.append('</tr>')
        html.append('</table>')
        return '\n'.join(html)


class DimensionSpinBox(QWidget):
    """Кастомный виджет для выбора размерности с кнопками-стрелками."""
    valueChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 3
        self._min = 2
        self._max = 15
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.btn_down = QPushButton("▼")
        self.btn_down.setFixedSize(36, 40)
        self.btn_down.setFont(QFont("Segoe UI", 10))
        self.btn_down.clicked.connect(self._decrease)
        layout.addWidget(self.btn_down)

        self.value_label = QLabel(str(self._value))
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setMinimumWidth(45)
        self.value_label.setMinimumHeight(40)
        self.value_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.value_label.setStyleSheet("""
            QLabel {
                background-color: #363636;
                color: #e0e0e0;
                border-top: 1px solid #3a3a3a;
                border-bottom: 1px solid #3a3a3a;
            }
        """)
        layout.addWidget(self.value_label)

        self.btn_up = QPushButton("▲")
        self.btn_up.setFixedSize(36, 40)
        self.btn_up.setFont(QFont("Segoe UI", 10))
        self.btn_up.clicked.connect(self._increase)
        layout.addWidget(self.btn_up)

        self._update_buttons()

        self.setStyleSheet("""
            QPushButton { background-color: #454545; color: #e0e0e0; border: none; border-radius: 8px; font-weight: bold; }
            QPushButton:hover { background-color: #5c9edd; }
            QPushButton:pressed { background-color: #4a8bc7; }
            QPushButton:disabled { background-color: #353535; color: #555555; }
        """)

    def _update_buttons(self):
        self.btn_down.setEnabled(self._value > self._min)
        self.btn_up.setEnabled(self._value < self._max)
        self.value_label.setText(str(self._value))

    def _increase(self):
        if self._value < self._max:
            self._value += 1
            self._update_buttons()
            self.valueChanged.emit(self._value)

    def _decrease(self):
        if self._value > self._min:
            self._value -= 1
            self._update_buttons()
            self.valueChanged.emit(self._value)

    def value(self) -> int: return self._value
    def setValue(self, value: int):
        self._value = max(self._min, min(self._max, value))
        self._update_buttons()
        self.valueChanged.emit(self._value)


class MatrixTableWidget(QTableWidget):
    """Виджет таблицы для ввода матрицы коэффициентов и вектора свободных членов."""
    def __init__(self, rows: int = 3, cols: int = 4, parent=None):
        super().__init__(rows, cols, parent)
        self.rows_count = rows
        self.cols_count = cols
        self._setup_ui()

    def _setup_ui(self):
        self.setFont(QFont("Consolas", 11))
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setDefaultSectionSize(80)
        self.verticalHeader().setDefaultSectionSize(35)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self._update_headers()

        self.setStyleSheet("""
            QTableWidget { border: none; border-radius: 12px; background-color: #2d2d2d; gridline-color: #3a3a3a; }
            QTableWidget::item { padding: 5px; border-radius: 4px; background-color: #2d2d2d; }
            QTableWidget::item:selected { background-color: #5c9edd; color: white; }
            QHeaderView { background-color: transparent; }
            QHeaderView::section { background-color: #2d2d2d; color: #707070; padding: 4px 8px; border: none; border-bottom: 1px solid #3a3a3a; border-right: 1px solid #3a3a3a; font-weight: normal; font-size: 11px; }
            QHeaderView::section:hover { background-color: #353535; color: #909090; }
            QTableCornerButton::section { background-color: #2d2d2d; border: none; }
        """)

    def _update_headers(self):
        n = self.rows_count
        self.setHorizontalHeaderLabels([f"x{i+1}" for i in range(n)] + ["b"])
        self.setVerticalHeaderLabels([str(i+1) for i in range(n)])

    def set_dimension(self, n: int):
        self.rows_count = n
        self.cols_count = n + 1
        self.setRowCount(n)
        self.setColumnCount(n + 1)
        self._update_headers()
        self.clearContents()

    def get_matrix(self) -> Optional[np.ndarray]:
        n = self.rows_count
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                item = self.item(i, j)
                if item is None or item.text().strip() == "": return None
                try: A[i, j] = float(item.text().replace(",", "."))
                except ValueError: return None
        return A

    def get_vector_b(self) -> Optional[np.ndarray]:
        n = self.rows_count
        b = np.zeros(n)
        for i in range(n):
            item = self.item(i, n)
            if item is None or item.text().strip() == "": return None
            try: b[i] = float(item.text().replace(",", "."))
            except ValueError: return None
        return b

    def set_matrix(self, A: np.ndarray, b: np.ndarray):
        n = A.shape[0]
        self.set_dimension(n)
        for i in range(n):
            for j in range(n):
                item = QTableWidgetItem(f"{A[i, j]:.4f}")
                self.setItem(i, j, item)
            item_b = QTableWidgetItem(f"{b[i]:.4f}")
            self.setItem(i, n, item_b)

    def clear_contents(self):
        self.clearContents()

    def fill_random(self, min_val: float = -10.0, max_val: float = 10.0):
        n = self.rows_count
        for i in range(n):
            for j in range(n + 1):
                val = np.random.uniform(min_val, max_val)
                item = QTableWidgetItem(f"{round(val, 2):.2f}")
                self.setItem(i, j, item)

    def fill_zeros(self):
        n = self.rows_count
        for i in range(n):
            for j in range(n + 1):
                self.setItem(i, j, QTableWidgetItem("0.00"))


class MainWindow(QMainWindow):
    """Главное окно приложения для решения СЛАУ методом Гаусса."""

    def __init__(self):
        super().__init__()
        self.solver = GaussSolver()
        self.formatter = StepsFormatter()
        self.last_result: Optional[SolutionResult] = None
        self._setup_window()
        self._setup_ui()
        self._setup_connections()
        self._apply_style()

    def _setup_window(self):
        self.setWindowTitle("Решение СЛАУ методом Гаусса")
        self.setMinimumSize(1200, 800)
        self.resize(1300, 850)
        self._set_windows_dark_title_bar()

    def _set_windows_dark_title_bar(self, widget=None):
        if platform.system() != "Windows": return
        target = widget if widget is not None else self
        try:
            hwnd = int(target.winId())
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            value = ctypes.c_int(1)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(value), ctypes.sizeof(value))
        except Exception:
            try:
                hwnd = int(target.winId())
                DWMWA_USE_IMMERSIVE_DARK_MODE_OLD = 19
                value = ctypes.c_int(1)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE_OLD, ctypes.byref(value), ctypes.sizeof(value))
            except Exception: pass

    def _apply_dark_title_to_dialog(self, dialog):
        dialog.setStyleSheet("""
            QDialog { background-color: #2d2d2d; }
            QDialog QLabel { color: #e0e0e0; font-size: 13px; }
            QDialog QPushButton { background-color: #404040; color: #e0e0e0; border: none; border-radius: 8px; padding: 8px 20px; min-width: 80px; font-size: 12px; }
            QDialog QPushButton:hover { background-color: #5c9edd; }
            QDialog QPushButton:pressed { background-color: #4a8bc7; }
            QFileDialog QTreeView, QFileDialog QListView { background-color: #252525; color: #e0e0e0; border: 1px solid #353535; border-radius: 6px; }
            QFileDialog QTreeView::item:selected, QFileDialog QListView::item:selected { background-color: #5c9edd; }
            QFileDialog QComboBox { background-color: #353535; color: #e0e0e0; border: 1px solid #454545; border-radius: 6px; padding: 4px 8px; }
            QFileDialog QLineEdit { background-color: #353535; color: #e0e0e0; border: 1px solid #454545; border-radius: 6px; padding: 4px 8px; }
            QFileDialog QSidebar { background-color: #252525; }
        """)
        QTimer.singleShot(10, lambda: self._set_windows_dark_title_bar(dialog))

    def _show_message_box(self, icon, title, text):
        msg_box = QMessageBox(self)
        msg_box.setIcon(icon)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.setStyleSheet("""
            QMessageBox { background-color: #2d2d2d; }
            QMessageBox QLabel { color: #e0e0e0; font-size: 13px; min-width: 300px; }
            QMessageBox QPushButton { background-color: #404040; color: #e0e0e0; border: none; border-radius: 8px; padding: 8px 24px; font-size: 12px; }
            QMessageBox QPushButton:hover { background-color: #5c9edd; }
        """)
        QTimer.singleShot(10, lambda: self._set_windows_dark_title_bar(msg_box))
        return msg_box.exec_()

    def _show_warning(self, title, text): return self._show_message_box(QMessageBox.Warning, title, text)
    def _show_error(self, title, text): return self._show_message_box(QMessageBox.Critical, title, text)
    def _show_info(self, title, text): return self._show_message_box(QMessageBox.Information, title, text)

    def _get_open_file_name(self, caption, directory, filter_str):
        dialog = QFileDialog(self, caption, directory, filter_str)
        dialog.setFileMode(QFileDialog.ExistingFile)
        self._apply_dark_title_to_dialog(dialog)
        if dialog.exec_() == QFileDialog.Accepted: return dialog.selectedFiles()[0] if dialog.selectedFiles() else ""
        return ""

    def _get_save_file_name(self, caption, directory, filter_str):
        dialog = QFileDialog(self, caption, directory, filter_str)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        self._apply_dark_title_to_dialog(dialog)
        if dialog.exec_() == QFileDialog.Accepted: return dialog.selectedFiles()[0] if dialog.selectedFiles() else ""
        return ""

    def _setup_ui(self):
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel)

        splitter = QSplitter(Qt.Horizontal)
        left_panel = self._create_input_panel()
        splitter.addWidget(left_panel)

        right_panel = self._create_output_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([500, 700])
        main_layout.addWidget(splitter, 1)

    def _create_control_panel(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("controlPanel")
        frame.setMinimumHeight(105)
        frame.setMaximumHeight(115)

        layout = QHBoxLayout(frame)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 0, 20, 0)

        dim_group = QGroupBox("Размерность")
        dim_layout = QHBoxLayout(dim_group)
        dim_layout.setSpacing(10)
        dim_label = QLabel("n =")
        dim_layout.addWidget(dim_label)
        self.dim_spinbox = DimensionSpinBox()
        dim_layout.addWidget(self.dim_spinbox)
        layout.addWidget(dim_group)

        actions_group = QGroupBox("Действия")
        actions_layout = QHBoxLayout(actions_group)
        self.btn_clear = QPushButton("Очистить")
        self.btn_zeros = QPushButton("Нули")
        self.btn_random = QPushButton("Случайно")
        for btn in (self.btn_clear, self.btn_zeros, self.btn_random):
            btn.setMinimumHeight(40)
            actions_layout.addWidget(btn)
        layout.addWidget(actions_group)

        file_group = QGroupBox("Файл")
        file_layout = QHBoxLayout(file_group)
        self.btn_load = QPushButton("Загрузить")
        self.btn_save_input = QPushButton("Сохранить")
        for btn in (self.btn_load, self.btn_save_input):
            btn.setMinimumHeight(40)
            file_layout.addWidget(btn)
        layout.addWidget(file_group)

        layout.addStretch()

        self.btn_solve = QPushButton("Решить систему")
        self.btn_solve.setMinimumWidth(160)
        self.btn_solve.setMinimumHeight(60)
        self.btn_solve.setFont(QFont("Segoe UI", 13, QFont.Bold))
        layout.addWidget(self.btn_solve)

        return frame

    def _create_input_panel(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("inputPanel")
        layout = QVBoxLayout(frame)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Ввод данных")
        title.setObjectName("panelTitle")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        info_label = QLabel("Введите коэффициенты матрицы A и вектор b")
        info_label.setStyleSheet("color: #808080; font-size: 12px;")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)

        self.matrix_table = MatrixTableWidget(3, 4)
        layout.addWidget(self.matrix_table, 1)

        return frame

    def _create_output_panel(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("outputPanel")
        layout = QVBoxLayout(frame)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Результаты")
        title.setObjectName("panelTitle")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        solution_group = QGroupBox("Решение")
        solution_layout = QVBoxLayout(solution_group)
        self.solution_text = QTextBrowser()
        self.solution_text.setReadOnly(True)
        self.solution_text.setMinimumHeight(150)
        self.solution_text.setOpenExternalLinks(False)
        solution_layout.addWidget(self.solution_text)
        layout.addWidget(solution_group, 1)

        steps_group = QGroupBox("Пошаговое решение")
        steps_layout = QVBoxLayout(steps_group)
        self.steps_text = QTextBrowser()
        self.steps_text.setReadOnly(True)
        self.steps_text.setOpenExternalLinks(False)
        steps_layout.addWidget(self.steps_text)
        layout.addWidget(steps_group, 2)

        self.btn_save_result = QPushButton("Сохранить результат")
        self.btn_save_result.setMinimumWidth(220)
        self.btn_save_result.setMinimumHeight(42)
        layout.addWidget(self.btn_save_result, alignment=Qt.AlignCenter)

        return frame

    def _setup_connections(self):
        self.dim_spinbox.valueChanged.connect(self._on_dimension_changed)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_zeros.clicked.connect(self._on_fill_zeros)
        self.btn_random.clicked.connect(self._on_fill_random)
        self.btn_load.clicked.connect(self._on_load_from_file)
        self.btn_save_input.clicked.connect(self._on_save_input)
        self.btn_save_result.clicked.connect(self._on_save_result)
        self.btn_solve.clicked.connect(self._on_solve)

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            #centralWidget { background-color: #1e1e1e; }
            QFrame#controlPanel, QFrame#inputPanel, QFrame#outputPanel { background-color: #2d2d2d; border-radius: 16px; border: 1px solid #404040; }
            QLabel#panelTitle { color: #e0e0e0; padding: 8px; }
            QGroupBox { font-weight: bold; font-size: 12px; color: #909090; border: 1px solid #3a3a3a; border-radius: 10px; margin-top: 14px; padding-top: 14px; background-color: #323232; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 8px; background-color: #323232; border-radius: 4px; }
            QPushButton { background-color: #404040; color: #e0e0e0; border: none; border-radius: 10px; padding: 12px 20px; font-size: 12px; font-weight: 500; }
            QPushButton:hover { background-color: #505050; }
            QPushButton:pressed { background-color: #383838; }
            QTextBrowser, QTextEdit { background-color: #252525; border: 1px solid #353535; border-radius: 12px; padding: 12px; color: #e0e0e0; selection-background-color: #5c9edd; font-size: 12px; }
            QScrollBar:vertical { background-color: #252525; width: 14px; border-radius: 7px; margin: 0px; }
            QScrollBar::handle:vertical { background-color: #454545; border-radius: 5px; min-height: 30px; margin: 2px; }
            QScrollBar::handle:vertical:hover { background-color: #5c9edd; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { background-color: none; height: 0px; }
            QScrollBar:horizontal { background-color: #252525; height: 14px; border-radius: 7px; margin: 0px; }
            QScrollBar::handle:horizontal { background-color: #454545; border-radius: 5px; min-width: 30px; margin: 2px; }
            QScrollBar::handle:horizontal:hover { background-color: #5c9edd; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { background-color: none; width: 0px; }
        """)

        self.btn_solve.setStyleSheet("""
            QPushButton { background: #4caf50; color: white; font-size: 14px; font-weight: bold; border-radius: 12px; }
            QPushButton:hover { background: #5cbf60; }
            QPushButton:pressed { background: #388e3c; }
        """)

    def _on_dimension_changed(self, value: int):
        self.matrix_table.set_dimension(value)
        self.solution_text.clear()
        self.steps_text.clear()
        self.statusBar().showMessage(f"Размерность изменена на {value}×{value}")

    def _on_clear(self):
        self.matrix_table.clear_contents()
        self.solution_text.clear()
        self.steps_text.clear()
        self.statusBar().showMessage("Таблица очищена")

    def _on_fill_zeros(self):
        self.matrix_table.fill_zeros()
        self.statusBar().showMessage("Таблица заполнена нулями")

    def _on_fill_random(self):
        self.matrix_table.fill_random()
        self.statusBar().showMessage("Таблица заполнена случайными значениями")

    def _on_load_from_file(self):
        file_path = self._get_open_file_name("Загрузить данные", str(Path.home()), "Excel файлы (*.xlsx);;Текстовые файлы (*.txt);;Все файлы (*.*)")
        if not file_path: return
        try:
            data =[]
            if file_path.endswith('.xlsx'):
                try:
                    import openpyxl
                    wb = openpyxl.load_workbook(file_path, data_only=True)
                    sheet = wb.active
                    for row in sheet.iter_rows(values_only=True):
                        clean_row =[]
                        for v in row:
                            if v is not None and str(v).strip() != '':
                                try: clean_row.append(float(str(v).replace(',', '.')))
                                except ValueError: continue
                        if clean_row: data.append(clean_row)
                except ImportError:
                    self._show_error("Ошибка", "Установите openpyxl: pip install openpyxl")
                    return
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            data.append([float(v) for v in line.replace(',', ' ').split()])

            if not data: raise ValueError("Файл не содержит числовых данных")
            n = len(data)
            expected_cols = n + 1
            for i, row in enumerate(data):
                if len(row) != expected_cols: raise ValueError(f"Строка {i + 1}: ожидается {expected_cols} значений")

            A = np.array([row[:-1] for row in data])
            b = np.array([row[-1] for row in data])

            self.dim_spinbox.setValue(n)
            self.matrix_table.set_matrix(A, b)
            self.solution_text.clear()
            self.steps_text.clear()
            self.statusBar().showMessage(f"Данные загружены из файла: {Path(file_path).name}")

        except Exception as e:
            self._show_error("Ошибка", str(e))

    def _on_save_input(self):
        A = self.matrix_table.get_matrix()
        b = self.matrix_table.get_vector_b()
        if A is None or b is None: return self._show_warning("Предупреждение", "Не все ячейки заполнены.")

        file_path = self._get_save_file_name("Сохранить данные", str(Path.home() / "matrix_data.xlsx"), "Excel файлы (*.xlsx);;Текстовые файлы (*.txt);;Все файлы (*.*)")
        if not file_path: return

        try:
            n = A.shape[0]
            if file_path.endswith('.xlsx'):
                try:
                    import openpyxl
                    wb = openpyxl.Workbook()
                    ws = wb.active
                    ws.append([f"x{i + 1}" for i in range(n)] + ["b"])
                    for i in range(n):
                        ws.append([float(A[i, j]) for j in range(n)] +[float(b[i])])
                    wb.save(file_path)
                except ImportError:
                    return self._show_error("Ошибка", "Установите openpyxl: pip install openpyxl")
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    for i in range(n):
                        f.write(" ".join(f"{A[i, j]:12.6f}" for j in range(n)) + f" {b[i]:12.6f}\n")

            self.statusBar().showMessage(f"Данные сохранены: {Path(file_path).name}")
        except Exception as e:
            self._show_error("Ошибка", str(e))

    def _on_save_result(self):
        if self.last_result is None: return self._show_warning("Предупреждение", "Сначала решите систему.")
        file_path = self._get_save_file_name("Сохранить результат", str(Path.home() / "solution_result.xlsx"), "Excel файлы (*.xlsx);;Текстовые файлы (*.txt);;Все файлы (*.*)")
        if not file_path: return

        try:
            if file_path.endswith('.xlsx'):
                try:
                    wb = openpyxl.Workbook()
                    ws_res = wb.active
                    ws_res.title = "Результат"
                    ws_res.append(["Отчет о решении СЛАУ методом Гаусса"])
                    ws_res['A1'].font = Font(bold=True, size=14)
                    ws_res.append([])
                    ws_res.append(["Статус решения:", self.last_result.status.value])
                    ws_res.append(["Ранг матрицы:", self.last_result.rank])
                    ws_res.append([])
                    if self.last_result.solution is not None:
                        ws_res.append(["Найденное решение:"])
                        for i, val in enumerate(self.last_result.solution):
                            ws_res.append([f"x[{i + 1}]", float(val)])
                    ws_res.column_dimensions['A'].width = 25
                    ws_res.column_dimensions['B'].width = 35

                    ws_steps = wb.create_sheet("Пошаговое решение")
                    ws_steps.append(["Шаги решения метода Гаусса"])
                    ws_steps.append([])
                    for step in self.last_result.steps: ws_steps.append([step])
                    ws_steps.column_dimensions['A'].width = 90
                    wb.save(file_path)
                except ImportError: return self._show_error("Ошибка", "Установите openpyxl")
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("РЕШЕНИЕ СИСТЕМЫ\n" + "=" * 60 + "\n\n")
                    f.write(f"Статус: {self.last_result.status.value}\n")
                    f.write(f"Ранг матрицы: {self.last_result.rank}\n\n")
                    if self.last_result.solution is not None:
                        for i, val in enumerate(self.last_result.solution):
                            f.write(f"x[{i + 1}] = {val:.6f}\n")
                    f.write("\nПОШАГОВОЕ РЕШЕНИЕ:\n" + "-" * 60 + "\n")
                    for step in self.last_result.steps:
                        f.write(step + "\n")

            self.statusBar().showMessage(f"Результат сохранен: {Path(file_path).name}")
        except Exception as e:
            self._show_error("Ошибка", str(e))

    def _on_solve(self):
        A = self.matrix_table.get_matrix()
        b = self.matrix_table.get_vector_b()
        if A is None or b is None: return self._show_warning("Предупреждение", "Не все ячейки заполнены корректными числами.")
        try:
            self.last_result = self.solver.solve(A, b)
            self._display_result()
            self.statusBar().showMessage(f"Статус: {self.last_result.status.value}")
        except Exception as e:
            self._show_error("Ошибка", str(e))

    def _display_result(self):
        if self.last_result is None: return
        result = self.last_result
        solution_html = [HTML_STYLES, "<body>"]

        if result.status == SolutionStatus.UNIQUE_SOLUTION:
            solution_html.append('<div style="font-size: 14px; font-weight: bold; color: #4caf50; margin-bottom: 12px;">Единственное решение:</div>')
            if result.solution is not None:
                for i, val in enumerate(result.solution):
                    solution_html.append(f'<div class="solution-line" style="margin-left: 0;">x<sub>{i+1}</sub> = {val:.6f}</div>')

            A = self.matrix_table.get_matrix()
            b = self.matrix_table.get_vector_b()
            if A is not None and b is not None and result.solution is not None:
                is_correct, residual, max_error = GaussSolver.verify_solution(A, b, result.solution)
                solution_html.append(
                    '<div style="margin-top: 15px; border-top: 1px solid #454545; padding-top: 10px;">'
                    '<div style="color: #5c9edd; font-weight: bold; margin-bottom: 5px;">Проверка решения:</div>'
                    f'<div class="text-line">Максимальная невязка: <span style="font-family: Consolas;">{max_error:.2e}</span></div>'
                    f'<div class="text-line" style="color: {"#4caf50" if is_correct else "#ef5350"}; font-weight: bold;">{"Решение верно" if is_correct else "Решение неверно"}</div>'
                    '</div>'
                )
        elif result.status == SolutionStatus.NO_SOLUTION:
            solution_html.append('<div class="error-text" style="font-size: 14px; margin-bottom: 8px;">Система несовместна</div>')
            solution_html.append('<div class="text-line">Решений нет.</div>')
        elif result.status == SolutionStatus.INFINITE_SOLUTIONS:
            solution_html.append('<div class="warning-text" style="font-size: 14px; margin-bottom: 8px;">Бесконечное множество решений</div>')
            solution_html.append(f'<div class="text-line">Количество свободных переменных: {result.rank}</div>')

        solution_html.append(
            '<div style="margin-top: 15px; border-top: 1px solid #454545; padding-top: 10px;">'
            f'<div class="text-line" style="color: #909090;">Ранг матрицы: {result.rank}</div>'
            '</div>'
        )

        solution_html.append("</body>")
        self.solution_text.setHtml("\n".join(solution_html))

        steps_html = self.formatter.format_steps(result.steps)
        self.steps_text.setHtml(steps_html)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, QColor(224, 224, 224))
    palette.setColor(QPalette.Base, QColor(45, 45, 45))
    palette.setColor(QPalette.AlternateBase, QColor(54, 54, 54))
    palette.setColor(QPalette.ToolTipBase, QColor(224, 224, 224))
    palette.setColor(QPalette.ToolTipText, QColor(30, 30, 30))
    palette.setColor(QPalette.Text, QColor(224, 224, 224))
    palette.setColor(QPalette.Button, QColor(54, 54, 54))
    palette.setColor(QPalette.ButtonText, QColor(224, 224, 224))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Highlight, QColor(92, 158, 221))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    from PyQt5.QtCore import QLocale
    QLocale.setDefault(QLocale(QLocale.Russian))

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()